import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multitask_transformer_finetune import (
    BatchTargets,
    CSVLogger,
    build_regression_stats,
    collect_predictions,
    compute_loss,
    evaluate,
    infer_classification_targets,
    infer_numeric_targets,
    parse_csv_list,
    resolve_input_path,
    run_epoch,
    save_predictions,
    set_seed,
    train_val_split,
)


SMILES_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+\]|Br?|Cl?|Si?|Se?|Na|Li|Ca|Al|Mg|Zn|Sn|Ag|Au|Fe|As|@@?|=|#|-|\+|\\|/|\(|\)|\.|:|~|@|\?|>>?|\*|\$|%[0-9]{2}|[0-9]|[A-Za-z])"
)


def default_chemformer_paths() -> Tuple[Path, Path, Optional[Path]]:
    checkpoint_v2_candidates = [
        PROJECT_ROOT / "chemformer" / "model__v2.ckpt",
        PROJECT_ROOT / "chemformer" / "model_v2.ckpt",
        PROJECT_ROOT / "chemformer" / "step=1000000_v2.ckpt",
    ]
    checkpoint_v1_candidates = [
        PROJECT_ROOT / "chemformer" / "model.ckpt",
        PROJECT_ROOT / "chemformer" / "step=1000000.ckpt",
    ]
    hparams = PROJECT_ROOT / "chemformer" / "hparams.yaml"

    vocab_candidates = [
        PROJECT_ROOT / "chemformer" / "bart_vocab_downstream.json",
        PROJECT_ROOT / "chemformer" / "bart_vocab.json",
        PROJECT_ROOT / "bart_vocab_downstream.json",
        PROJECT_ROOT / "bart_vocab.json",
    ]
    vocab_path = next((path for path in vocab_candidates if path.exists()), None)

    checkpoint = next((path for path in checkpoint_v2_candidates if path.exists()), None)
    if checkpoint is None:
        checkpoint = next((path for path in checkpoint_v1_candidates if path.exists()), checkpoint_v1_candidates[0])
    return checkpoint, hparams, vocab_path


def resolve_special_token(token_to_id: Dict[str, int], candidates: Sequence[str]) -> Optional[str]:
    for token in candidates:
        if token in token_to_id:
            return token
    return None


def extract_vocab_mapping(raw_vocab: Dict[str, object]) -> Dict[str, int]:
    if "vocabulary" in raw_vocab and isinstance(raw_vocab["vocabulary"], list):
        return {str(token): idx for idx, token in enumerate(raw_vocab["vocabulary"])}

    candidate = raw_vocab
    visited = set()

    while isinstance(candidate, dict):
        marker = id(candidate)
        if marker in visited:
            break
        visited.add(marker)

        if candidate and all(isinstance(value, (int, float, str)) for value in candidate.values()):
            return candidate

        if "model" in candidate and isinstance(candidate["model"], dict):
            candidate = candidate["model"]
            continue
        if "vocab" in candidate and isinstance(candidate["vocab"], dict):
            candidate = candidate["vocab"]
            continue
        break

    raise ValueError(
        "Could not extract a flat Chemformer vocabulary mapping from the provided JSON. "
        "Expected token->id pairs, or a nested tokenizer JSON containing model.vocab."
    )


class RegexSmilesTokenizer:
    def __init__(self, vocab_path: Path, pad_token_idx: int) -> None:
        with open(vocab_path, "r", encoding="utf-8") as handle:
            vocab = json.load(handle)

        if not isinstance(vocab, dict):
            raise ValueError(f"Expected {vocab_path} to contain a JSON object vocabulary.")
        vocab = extract_vocab_mapping(vocab)

        if all(str(key).isdigit() for key in vocab.keys()):
            self.id_to_token = {int(key): str(value) for key, value in vocab.items()}
            self.token_to_id = {token: idx for idx, token in self.id_to_token.items()}
        else:
            self.token_to_id = {str(key): int(value) for key, value in vocab.items()}
            self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        if pad_token_idx not in self.id_to_token:
            raise ValueError(
                f"Pad token index {pad_token_idx} was not found in vocabulary {vocab_path}."
            )

        self.pad_token_idx = pad_token_idx
        self.pad_token = self.id_to_token[pad_token_idx]
        self.unk_token = resolve_special_token(self.token_to_id, ["<unk>", "[UNK]", "?", "<UNK>"])
        self.bos_token = resolve_special_token(self.token_to_id, ["<s>", "^", "[CLS]", "<bos>", "<BOS>"])
        self.eos_token = resolve_special_token(self.token_to_id, ["</s>", "&", "[SEP]", "<eos>", "<EOS>"])

    def tokenize(self, smiles: str) -> List[str]:
        tokens: List[str] = []
        cursor = 0
        for match in SMILES_TOKEN_PATTERN.finditer(smiles):
            if match.start() > cursor:
                tokens.extend(list(smiles[cursor:match.start()]))
            tokens.append(match.group(0))
            cursor = match.end()
        if cursor < len(smiles):
            tokens.extend(list(smiles[cursor:]))
        return tokens

    def encode(self, smiles: str, max_length: int) -> Tuple[Tensor, Tensor]:
        tokens = self.tokenize(smiles)
        if self.bos_token is not None:
            tokens = [self.bos_token] + tokens
        if self.eos_token is not None:
            tokens = tokens + [self.eos_token]

        token_ids: List[int] = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            elif self.unk_token is not None:
                token_ids.append(self.token_to_id[self.unk_token])
            else:
                raise KeyError(
                    f"Token '{token}' was not found in the Chemformer vocabulary. "
                    "Provide the matching Chemformer vocab JSON."
                )

        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            if self.eos_token is not None:
                token_ids[-1] = self.token_to_id[self.eos_token]

        attention_mask = [1] * len(token_ids)
        padding = max_length - len(token_ids)
        if padding > 0:
            token_ids.extend([self.pad_token_idx] * padding)
            attention_mask.extend([0] * padding)

        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)


class ChemformerDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: RegexSmilesTokenizer,
        smiles_column: str,
        regression_targets: Sequence[str],
        classification_targets: Sequence[str],
        regression_means: Dict[str, float],
        regression_stds: Dict[str, float],
        max_length: int,
    ) -> None:
        self.smiles = frame[smiles_column].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.regression_targets = list(regression_targets)
        self.classification_targets = list(classification_targets)

        regression_values = []
        regression_mask = []
        for column in self.regression_targets:
            raw_series = frame[column] if column in frame.columns else pd.Series(float("nan"), index=frame.index)
            series = pd.to_numeric(raw_series, errors="coerce")
            mask = series.notna().astype("float32").to_numpy()
            values = series.fillna(regression_means[column]).to_numpy(dtype="float32")
            values = (values - regression_means[column]) / regression_stds[column]
            regression_values.append(values)
            regression_mask.append(mask)

        classification_values = []
        classification_mask = []
        for column in self.classification_targets:
            raw_series = frame[column] if column in frame.columns else pd.Series(float("nan"), index=frame.index)
            series = pd.to_numeric(raw_series, errors="coerce")
            mask = series.notna().astype("float32").to_numpy()
            values = series.fillna(0.0).to_numpy(dtype="float32")
            classification_values.append(values)
            classification_mask.append(mask)

        self.regression_values = (
            torch.tensor(list(zip(*regression_values)), dtype=torch.float32)
            if regression_values
            else torch.zeros((len(frame), 0), dtype=torch.float32)
        )
        self.regression_mask = (
            torch.tensor(list(zip(*regression_mask)), dtype=torch.float32)
            if regression_mask
            else torch.zeros((len(frame), 0), dtype=torch.float32)
        )
        self.classification_values = (
            torch.tensor(list(zip(*classification_values)), dtype=torch.float32)
            if classification_values
            else torch.zeros((len(frame), 0), dtype=torch.float32)
        )
        self.classification_mask = (
            torch.tensor(list(zip(*classification_mask)), dtype=torch.float32)
            if classification_mask
            else torch.zeros((len(frame), 0), dtype=torch.float32)
        )

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        input_ids, attention_mask = self.tokenizer.encode(self.smiles[index], self.max_length)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "regression_values": self.regression_values[index],
            "regression_mask": self.regression_mask[index],
            "classification_values": self.classification_values[index],
            "classification_mask": self.classification_mask[index],
        }


def activation_fn(name: str):
    if name == "gelu":
        return nn.functional.gelu
    if name == "relu":
        return nn.functional.relu
    raise ValueError(f"Unsupported Chemformer activation: {name}")


class PreNormEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_feedforward: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation_fn(activation)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        del is_causal
        src_norm = self.norm1(src)
        attn_output = self.self_attn(
            src_norm,
            src_norm,
            src_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )[0]
        src = src + self.dropout1(attn_output)
        ff_input = self.norm2(src)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(ff_input))))
        src = src + self.dropout2(ff_output)
        return src


class PreNormDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_feedforward: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation_fn(activation)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        del tgt_is_causal, memory_is_causal
        tgt_norm = self.norm1(tgt)
        self_attn_output = self.self_attn(
            tgt_norm,
            tgt_norm,
            tgt_norm,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout1(self_attn_output)

        tgt_norm = self.norm2(tgt)
        cross_attn_output = self.multihead_attn(
            tgt_norm,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout2(cross_attn_output)

        ff_input = self.norm3(tgt)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(ff_input))))
        tgt = tgt + self.dropout3(ff_output)
        return tgt


class NativeChemformerBackbone(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_feedforward: int,
        max_seq_len: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocabulary_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(max_seq_len, d_model))
        self.encoder = nn.TransformerEncoder(
            PreNormEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation),
            num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.decoder = nn.TransformerDecoder(
            PreNormDecoderLayer(d_model, num_heads, d_feedforward, dropout, activation),
            num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.token_fc = nn.Linear(d_model, vocabulary_size)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def _construct_input(self, token_ids: Tensor) -> Tensor:
        token_embeddings = self.emb(token_ids) * math.sqrt(self.d_model)
        seq_len = token_embeddings.size(0)
        positional_embeddings = self.pos_emb[:seq_len].unsqueeze(1)
        return self.dropout(token_embeddings + positional_embeddings)

    def encode(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        token_ids = input_ids.transpose(0, 1)
        encoder_pad_mask = attention_mask == 0
        encoder_embeddings = self._construct_input(token_ids)
        encoded = self.encoder(encoder_embeddings, src_key_padding_mask=encoder_pad_mask)
        return encoded.transpose(0, 1)


class ChemformerPropertyModel(nn.Module):
    def __init__(
        self,
        checkpoint_path: Path,
        hparams: Dict[str, object],
        regression_targets: Sequence[str],
        classification_targets: Sequence[str],
        mlp_hidden_dims: Sequence[int],
        head_dropout: float,
    ) -> None:
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        vocabulary_size = int(hparams.get("vocabulary_size", hparams.get("vocab_size")))
        d_model = int(hparams["d_model"])
        num_layers = int(hparams["num_layers"])
        num_heads = int(hparams["num_heads"])
        d_feedforward = int(hparams["d_feedforward"])
        max_seq_len = int(hparams["max_seq_len"])
        dropout = float(hparams.get("dropout", 0.1))
        activation = str(hparams.get("activation", "gelu"))

        self.backbone = NativeChemformerBackbone(
            vocabulary_size=vocabulary_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_feedforward=d_feedforward,
            max_seq_len=max_seq_len,
            dropout=dropout,
            activation=activation,
        )
        missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
        if missing_keys:
            raise ValueError(f"Missing Chemformer checkpoint keys: {missing_keys}")
        if unexpected_keys:
            raise ValueError(f"Unexpected Chemformer checkpoint keys: {unexpected_keys}")

        layers: List[nn.Module] = []
        input_dim = d_model
        for dim in mlp_hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(head_dropout))
            input_dim = dim
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.regression_head = nn.Linear(input_dim, len(regression_targets)) if regression_targets else None
        self.classification_head = (
            nn.Linear(input_dim, len(classification_targets)) if classification_targets else None
        )

    @staticmethod
    def mean_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
        masked_hidden = hidden_states * mask
        summed = masked_hidden.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Dict[str, Tensor]:
        encoder_hidden = self.backbone.encode(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(encoder_hidden, attention_mask)
        features = self.mlp(pooled)
        regression_logits = (
            self.regression_head(features) if self.regression_head is not None else features.new_zeros((features.size(0), 0))
        )
        classification_logits = (
            self.classification_head(features)
            if self.classification_head is not None
            else features.new_zeros((features.size(0), 0))
        )
        return {
            "regression_logits": regression_logits,
            "classification_logits": classification_logits,
        }


def parse_args() -> argparse.Namespace:
    default_checkpoint, default_hparams, default_vocab = default_chemformer_paths()
    parser = argparse.ArgumentParser(
        description="Fine-tune a native Chemformer checkpoint with an MLP head for property prediction."
    )
    parser.add_argument("--train-csv", type=Path, default=PROJECT_ROOT / "Training_and_Eval_Dataset.csv")
    parser.add_argument("--test-csv", type=Path, default=PROJECT_ROOT / "Test_dataset.csv")
    parser.add_argument("--smiles-column", type=str, default="SMILES")
    parser.add_argument("--target-columns", type=str, default="")
    parser.add_argument("--classification-targets", type=str, default="")
    parser.add_argument("--checkpoint-path", type=Path, default=default_checkpoint)
    parser.add_argument("--hparams-yaml", type=Path, default=default_hparams)
    parser.add_argument("--vocab-path", type=Path, default=default_vocab)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "chemformer" / "outputs")
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--mlp-hidden-dims", type=str, default="512,256")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--regression-loss-weight", type=float, default=1.0)
    parser.add_argument("--classification-loss-weight", type=float, default=1.0)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--no-test-predictions", action="store_true")
    return parser.parse_args()


def load_hparams(hparams_path: Path) -> Dict[str, object]:
    with open(hparams_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected {hparams_path} to contain a YAML mapping.")
    return config


def train_and_predict(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.train_csv = resolve_input_path(args.train_csv)
    args.test_csv = resolve_input_path(args.test_csv)
    args.checkpoint_path = resolve_input_path(args.checkpoint_path)
    args.hparams_yaml = resolve_input_path(args.hparams_yaml)
    args.vocab_path = resolve_input_path(args.vocab_path) if args.vocab_path is not None else None

    if args.vocab_path is None or not args.vocab_path.exists():
        raise FileNotFoundError(
            "Chemformer vocabulary JSON was not found. Download `bart_vocab.json` or "
            "`bart_vocab_downstream.json` from the official Chemformer repo and pass it via --vocab-path."
        )

    hparams = load_hparams(args.hparams_yaml)
    if "vocabulary_size" not in hparams and "vocab_size" in hparams:
        hparams["vocabulary_size"] = hparams["vocab_size"]

    frame = pd.read_csv(args.train_csv)
    if args.smiles_column not in frame.columns:
        raise ValueError(f"SMILES column '{args.smiles_column}' was not found in {args.train_csv}.")

    target_columns = parse_csv_list(args.target_columns) or infer_numeric_targets(frame, args.smiles_column)
    if not target_columns:
        raise ValueError("No numeric target columns were found. Use --target-columns to set them explicitly.")

    requested_classification = parse_csv_list(args.classification_targets)
    classification_targets = requested_classification or infer_classification_targets(frame, target_columns)
    regression_targets = [column for column in target_columns if column not in set(classification_targets)]

    tokenizer = RegexSmilesTokenizer(
        vocab_path=args.vocab_path,
        pad_token_idx=int(hparams["pad_token_idx"]),
    )
    train_frame, val_frame = train_val_split(frame, args.val_fraction, args.seed)
    regression_means, regression_stds = build_regression_stats(train_frame, regression_targets)
    max_length = int(args.max_length or hparams["max_seq_len"])

    train_dataset = ChemformerDataset(
        frame=train_frame,
        tokenizer=tokenizer,
        smiles_column=args.smiles_column,
        regression_targets=regression_targets,
        classification_targets=classification_targets,
        regression_means=regression_means,
        regression_stds=regression_stds,
        max_length=max_length,
    )
    val_dataset = ChemformerDataset(
        frame=val_frame,
        tokenizer=tokenizer,
        smiles_column=args.smiles_column,
        regression_targets=regression_targets,
        classification_targets=classification_targets,
        regression_means=regression_means,
        regression_stds=regression_stds,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    mlp_hidden_dims = [int(dim) for dim in parse_csv_list(args.mlp_hidden_dims)]
    model = ChemformerPropertyModel(
        checkpoint_path=args.checkpoint_path,
        hparams=hparams,
        regression_targets=regression_targets,
        classification_targets=classification_targets,
        mlp_hidden_dims=mlp_hidden_dims,
        head_dropout=args.head_dropout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    csv_logger = CSVLogger(
        args.output_dir / "training_log.csv",
        fieldnames=[
            "event",
            "epoch",
            "global_step",
            "step_in_epoch",
            "split",
            "loss",
            "train_loss",
            "val_loss",
            "metrics_json",
        ],
    )

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    checkpoint_path = args.output_dir / "best_model.pt"
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, global_step = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            regression_weight=args.regression_loss_weight,
            classification_weight=args.classification_loss_weight,
            epoch=epoch,
            global_step=global_step,
            log_every_steps=args.log_every_steps,
            csv_logger=csv_logger,
        )
        val_loss, global_step = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            regression_weight=args.regression_loss_weight,
            classification_weight=args.classification_loss_weight,
            epoch=epoch,
            global_step=global_step,
            log_every_steps=args.log_every_steps,
            csv_logger=None,
        )
        metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            regression_targets=regression_targets,
            classification_targets=classification_targets,
            regression_means=regression_means,
            regression_stds=regression_stds,
        )
        epoch_record: Dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        epoch_record.update(metrics)
        history.append(epoch_record)
        csv_logger.log(
            {
                "event": "epoch_summary",
                "epoch": epoch,
                "global_step": global_step,
                "step_in_epoch": "",
                "split": "val",
                "loss": "",
                "train_loss": train_loss,
                "val_loss": val_loss,
                "metrics_json": json.dumps(metrics, sort_keys=True),
            }
        )
        print(json.dumps(epoch_record), flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "checkpoint_path": str(args.checkpoint_path),
                    "hparams_yaml": str(args.hparams_yaml),
                    "vocab_path": str(args.vocab_path),
                    "smiles_column": args.smiles_column,
                    "regression_targets": regression_targets,
                    "classification_targets": classification_targets,
                    "regression_means": regression_means,
                    "regression_stds": regression_stds,
                    "max_length": max_length,
                    "mlp_hidden_dims": mlp_hidden_dims,
                    "head_dropout": args.head_dropout,
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )

    with open(args.output_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    with open(args.output_dir / "training_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "train_csv": str(args.train_csv),
                "test_csv": str(args.test_csv),
                "smiles_column": args.smiles_column,
                "target_columns": target_columns,
                "classification_targets": classification_targets,
                "regression_targets": regression_targets,
                "checkpoint_path": str(args.checkpoint_path),
                "hparams_yaml": str(args.hparams_yaml),
                "vocab_path": str(args.vocab_path),
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "head_dropout": args.head_dropout,
                "mlp_hidden_dims": mlp_hidden_dims,
                "max_length": max_length,
                "val_fraction": args.val_fraction,
                "seed": args.seed,
            },
            handle,
            indent=2,
        )

    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    if args.no_test_predictions:
        return

    test_frame = pd.read_csv(args.test_csv)
    if args.smiles_column not in test_frame.columns:
        raise ValueError(f"SMILES column '{args.smiles_column}' was not found in {args.test_csv}.")

    test_dataset = ChemformerDataset(
        frame=test_frame,
        tokenizer=tokenizer,
        smiles_column=args.smiles_column,
        regression_targets=regression_targets,
        classification_targets=classification_targets,
        regression_means=regression_means,
        regression_stds=regression_stds,
        max_length=max_length,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    predictions = collect_predictions(
        model=model,
        loader=test_loader,
        device=device,
        regression_targets=regression_targets,
        classification_targets=classification_targets,
        regression_means=regression_means,
        regression_stds=regression_stds,
    )
    save_predictions(
        source_frame=test_frame,
        output_path=args.output_dir / "test_predictions.csv",
        predictions=predictions,
        regression_targets=regression_targets,
        classification_targets=classification_targets,
    )


if __name__ == "__main__":
    train_and_predict(parse_args())
