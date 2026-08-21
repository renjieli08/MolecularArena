import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_input_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path
    return path


def infer_numeric_targets(frame: pd.DataFrame, smiles_column: str) -> List[str]:
    targets: List[str] = []
    for column in frame.columns:
        if column == smiles_column:
            continue
        numeric_series = pd.to_numeric(frame[column], errors="coerce")
        if numeric_series.notna().any():
            targets.append(column)
    return targets


def infer_classification_targets(
    frame: pd.DataFrame,
    candidate_targets: Sequence[str],
) -> List[str]:
    classification_targets: List[str] = []
    for column in candidate_targets:
        values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
        if len(values) == 0:
            continue
        if set(np.round(values, 6)).issubset({0.0, 1.0}):
            classification_targets.append(column)
    return classification_targets


def train_val_split(
    frame: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1.")

    indices = np.arange(len(frame))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_size = max(1, int(len(frame) * val_fraction))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    if len(train_indices) == 0:
        raise ValueError("Validation split consumed the full dataset. Reduce --val-fraction.")

    train_frame = frame.iloc[train_indices].reset_index(drop=True)
    val_frame = frame.iloc[val_indices].reset_index(drop=True)
    return train_frame, val_frame


def build_regression_stats(
    frame: pd.DataFrame,
    regression_targets: Sequence[str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for column in regression_targets:
        series = pd.to_numeric(frame[column], errors="coerce")
        mean = float(series.mean(skipna=True))
        std = float(series.std(skipna=True))
        if math.isnan(std) or std < 1e-8:
            std = 1.0
        means[column] = 0.0 if math.isnan(mean) else mean
        stds[column] = std
    return means, stds


class CSVLogger:
    def __init__(self, path: Path, fieldnames: Sequence[str], append: bool = False) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.path.exists()
        mode = "a" if append else "w"
        should_write_header = not append or not file_exists
        with open(self.path, mode, newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if should_write_header:
                writer.writeheader()

    def log(self, row: Dict[str, object]) -> None:
        record = {field: row.get(field, "") for field in self.fieldnames}
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(record)


@dataclass
class BatchTargets:
    regression_values: torch.Tensor
    regression_mask: torch.Tensor
    classification_values: torch.Tensor
    classification_mask: torch.Tensor


class MoleculeDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer,
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
            raw_series = frame[column] if column in frame.columns else pd.Series(np.nan, index=frame.index)
            series = pd.to_numeric(raw_series, errors="coerce")
            mask = series.notna().astype(np.float32).to_numpy()
            values = series.fillna(regression_means[column]).to_numpy(dtype=np.float32)
            values = (values - regression_means[column]) / regression_stds[column]
            regression_values.append(values)
            regression_mask.append(mask)

        classification_values = []
        classification_mask = []
        for column in self.classification_targets:
            raw_series = frame[column] if column in frame.columns else pd.Series(np.nan, index=frame.index)
            series = pd.to_numeric(raw_series, errors="coerce")
            mask = series.notna().astype(np.float32).to_numpy()
            values = series.fillna(0.0).to_numpy(dtype=np.float32)
            classification_values.append(values)
            classification_mask.append(mask)

        self.regression_values = (
            np.stack(regression_values, axis=1) if regression_values else np.zeros((len(frame), 0), dtype=np.float32)
        )
        self.regression_mask = (
            np.stack(regression_mask, axis=1) if regression_mask else np.zeros((len(frame), 0), dtype=np.float32)
        )
        self.classification_values = (
            np.stack(classification_values, axis=1)
            if classification_values
            else np.zeros((len(frame), 0), dtype=np.float32)
        )
        self.classification_mask = (
            np.stack(classification_mask, axis=1)
            if classification_mask
            else np.zeros((len(frame), 0), dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.smiles[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["regression_values"] = torch.tensor(self.regression_values[index], dtype=torch.float32)
        item["regression_mask"] = torch.tensor(self.regression_mask[index], dtype=torch.float32)
        item["classification_values"] = torch.tensor(self.classification_values[index], dtype=torch.float32)
        item["classification_mask"] = torch.tensor(self.classification_mask[index], dtype=torch.float32)
        return item


class MolecularPropertyModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        regression_targets: Sequence[str],
        classification_targets: Sequence[str],
        dropout: float,
        mlp_hidden_dims: Sequence[int],
        trust_remote_code: bool,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.is_encoder_decoder = bool(getattr(self.encoder.config, "is_encoder_decoder", False))
        hidden_size = getattr(self.encoder.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(f"Could not read hidden_size from encoder config for {model_name}.")

        layers: List[nn.Module] = []
        input_dim = hidden_size
        for dim in mlp_hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = dim
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.num_regression_targets = len(regression_targets)
        self.num_classification_targets = len(classification_targets)
        self.regression_head = (
            nn.Linear(input_dim, self.num_regression_targets) if self.num_regression_targets > 0 else None
        )
        self.classification_head = (
            nn.Linear(input_dim, self.num_classification_targets) if self.num_classification_targets > 0 else None
        )

    @staticmethod
    def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        masked_hidden = last_hidden_state * mask
        summed = masked_hidden.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.is_encoder_decoder:
            encoder_outputs = self.encoder.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
            pooled = self.mean_pool(encoder_outputs.last_hidden_state, attention_mask)
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)
        features = self.mlp(pooled)
        regression_logits = (
            self.regression_head(features)
            if self.regression_head is not None
            else features.new_zeros((features.size(0), 0))
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


def masked_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if predictions.numel() == 0:
        return predictions.new_tensor(0.0)
    errors = (predictions - targets) ** 2
    masked = errors * mask
    return masked.sum() / mask.sum().clamp(min=1.0)


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    losses = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    masked = losses * mask
    return masked.sum() / mask.sum().clamp(min=1.0)


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch_targets: BatchTargets,
    regression_weight: float,
    classification_weight: float,
) -> torch.Tensor:
    regression_loss = masked_mse_loss(
        outputs["regression_logits"],
        batch_targets.regression_values,
        batch_targets.regression_mask,
    )
    classification_loss = masked_bce_loss(
        outputs["classification_logits"],
        batch_targets.classification_values,
        batch_targets.classification_mask,
    )
    return regression_weight * regression_loss + classification_weight * classification_loss


def make_batch_targets(batch: Dict[str, torch.Tensor], device: torch.device) -> BatchTargets:
    return BatchTargets(
        regression_values=batch["regression_values"].to(device),
        regression_mask=batch["regression_mask"].to(device),
        classification_values=batch["classification_values"].to(device),
        classification_mask=batch["classification_mask"].to(device),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    regression_weight: float,
    classification_weight: float,
    epoch: int,
    global_step: int,
    log_every_steps: int,
    csv_logger: Optional[CSVLogger],
) -> Tuple[float, int]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    running_loss = 0.0

    for step_in_epoch, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_targets = make_batch_targets(batch, device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = compute_loss(outputs, batch_targets, regression_weight, classification_weight)

        if training:
            loss.backward()
            optimizer.step()
            global_step += 1

        batch_loss = float(loss.detach().cpu().item())
        total_loss += batch_loss * input_ids.size(0)
        running_loss += batch_loss

        if training and csv_logger is not None:
            csv_logger.log(
                {
                    "event": "train_step",
                    "epoch": epoch,
                    "global_step": global_step,
                    "step_in_epoch": step_in_epoch,
                    "split": "train",
                    "loss": batch_loss,
                }
            )

        if training and global_step % log_every_steps == 0:
            average_loss = running_loss / log_every_steps
            print(
                f"epoch={epoch} global_step={global_step} step_in_epoch={step_in_epoch} "
                f"train_loss={batch_loss:.6f} avg_loss_{log_every_steps}={average_loss:.6f}",
                flush=True,
            )
            running_loss = 0.0

    return total_loss / max(len(loader.dataset), 1), global_step


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    regression_targets: Sequence[str],
    classification_targets: Sequence[str],
    regression_means: Dict[str, float],
    regression_stds: Dict[str, float],
) -> Dict[str, np.ndarray]:
    model.eval()
    regression_logits: List[np.ndarray] = []
    classification_logits: List[np.ndarray] = []
    regression_values: List[np.ndarray] = []
    regression_mask: List[np.ndarray] = []
    classification_values: List[np.ndarray] = []
    classification_mask: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            regression_logits.append(outputs["regression_logits"].cpu().numpy())
            classification_logits.append(outputs["classification_logits"].cpu().numpy())
            regression_values.append(batch["regression_values"].cpu().numpy())
            regression_mask.append(batch["regression_mask"].cpu().numpy())
            classification_values.append(batch["classification_values"].cpu().numpy())
            classification_mask.append(batch["classification_mask"].cpu().numpy())

    regression_pred = np.concatenate(regression_logits, axis=0) if regression_logits else np.zeros((0, 0), dtype=np.float32)
    classification_pred = (
        np.concatenate(classification_logits, axis=0) if classification_logits else np.zeros((0, 0), dtype=np.float32)
    )
    regression_true = np.concatenate(regression_values, axis=0) if regression_values else np.zeros((0, 0), dtype=np.float32)
    regression_true_mask = np.concatenate(regression_mask, axis=0) if regression_mask else np.zeros((0, 0), dtype=np.float32)
    classification_true = (
        np.concatenate(classification_values, axis=0) if classification_values else np.zeros((0, 0), dtype=np.float32)
    )
    classification_true_mask = (
        np.concatenate(classification_mask, axis=0) if classification_mask else np.zeros((0, 0), dtype=np.float32)
    )

    for index, column in enumerate(regression_targets):
        regression_pred[:, index] = regression_pred[:, index] * regression_stds[column] + regression_means[column]
        regression_true[:, index] = regression_true[:, index] * regression_stds[column] + regression_means[column]

    classification_prob = 1.0 / (1.0 + np.exp(-classification_pred))

    return {
        "regression_pred": regression_pred,
        "classification_prob": classification_prob,
        "regression_true": regression_true,
        "regression_mask": regression_true_mask,
        "classification_true": classification_true,
        "classification_mask": classification_true_mask,
    }


def regression_metrics(
    predictions: np.ndarray,
    truths: np.ndarray,
    mask: np.ndarray,
    targets: Sequence[str],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for index, column in enumerate(targets):
        valid = mask[:, index] > 0
        if not np.any(valid):
            continue
        errors = predictions[valid, index] - truths[valid, index]
        metrics[f"{column}_rmse"] = float(np.sqrt(np.mean(errors**2)))
        metrics[f"{column}_mae"] = float(np.mean(np.abs(errors)))
    return metrics


def classification_metrics(
    probabilities: np.ndarray,
    truths: np.ndarray,
    mask: np.ndarray,
    targets: Sequence[str],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for index, column in enumerate(targets):
        valid = mask[:, index] > 0
        if not np.any(valid):
            continue
        preds = (probabilities[valid, index] >= 0.5).astype(np.float32)
        metrics[f"{column}_accuracy"] = float(np.mean(preds == truths[valid, index]))
    return metrics


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    regression_targets: Sequence[str],
    classification_targets: Sequence[str],
    regression_means: Dict[str, float],
    regression_stds: Dict[str, float],
) -> Dict[str, float]:
    predictions = collect_predictions(
        model=model,
        loader=loader,
        device=device,
        regression_targets=regression_targets,
        classification_targets=classification_targets,
        regression_means=regression_means,
        regression_stds=regression_stds,
    )
    metrics = {}
    metrics.update(
        regression_metrics(
            predictions["regression_pred"],
            predictions["regression_true"],
            predictions["regression_mask"],
            regression_targets,
        )
    )
    metrics.update(
        classification_metrics(
            predictions["classification_prob"],
            predictions["classification_true"],
            predictions["classification_mask"],
            classification_targets,
        )
    )
    return metrics


def save_predictions(
    source_frame: pd.DataFrame,
    output_path: Path,
    predictions: Dict[str, np.ndarray],
    regression_targets: Sequence[str],
    classification_targets: Sequence[str],
) -> None:
    frame = source_frame.copy()
    for index, column in enumerate(regression_targets):
        frame[f"pred_{column}"] = predictions["regression_pred"][:, index]
    for index, column in enumerate(classification_targets):
        frame[f"pred_{column}_prob"] = predictions["classification_prob"][:, index]
        frame[f"pred_{column}_label"] = (predictions["classification_prob"][:, index] >= 0.5).astype(int)
    frame.to_csv(output_path, index=False)


def parse_args(
    default_model_name: str,
    default_output_dir: Path,
    default_trust_remote_code: bool,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a molecular transformer with an MLP head for multitask property prediction."
    )
    parser.add_argument("--train-csv", type=Path, default=PROJECT_ROOT / "Training_and_Eval_Dataset.csv")
    parser.add_argument("--test-csv", type=Path, default=PROJECT_ROOT / "Test_dataset.csv")
    parser.add_argument("--smiles-column", type=str, default="SMILES")
    parser.add_argument("--target-columns", type=str, default="")
    parser.add_argument("--classification-targets", type=str, default="")
    parser.add_argument("--model-name", type=str, default=default_model_name)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp-hidden-dims", type=str, default="512,256")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--regression-loss-weight", type=float, default=1.0)
    parser.add_argument("--classification-loss-weight", type=float, default=1.0)
    parser.add_argument("--trust-remote-code", action="store_true", default=default_trust_remote_code)
    parser.add_argument("--no-test-predictions", action="store_true")
    parser.add_argument("--log-every-steps", type=int, default=100)
    return parser.parse_args()


def train_and_predict(
    args: argparse.Namespace,
) -> None:
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.train_csv = resolve_input_path(args.train_csv)
    args.test_csv = resolve_input_path(args.test_csv)

    frame = pd.read_csv(args.train_csv)
    if args.smiles_column not in frame.columns:
        raise ValueError(f"SMILES column '{args.smiles_column}' was not found in {args.train_csv}.")

    target_columns = parse_csv_list(args.target_columns) or infer_numeric_targets(frame, args.smiles_column)
    if not target_columns:
        raise ValueError("No numeric target columns were found. Use --target-columns to set them explicitly.")
    missing_target_columns = [column for column in target_columns if column not in frame.columns]
    if missing_target_columns:
        raise ValueError(
            "These target columns are missing from the training CSV: "
            + ", ".join(missing_target_columns)
        )

    requested_classification = parse_csv_list(args.classification_targets)
    classification_targets = requested_classification or infer_classification_targets(frame, target_columns)
    missing_classification_targets = [column for column in classification_targets if column not in target_columns]
    if missing_classification_targets:
        raise ValueError(
            "Classification targets must also be listed in target columns: "
            + ", ".join(missing_classification_targets)
        )
    regression_targets = [column for column in target_columns if column not in set(classification_targets)]

    if not regression_targets and not classification_targets:
        raise ValueError("No valid regression or classification targets were selected.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    train_frame, val_frame = train_val_split(frame, args.val_fraction, args.seed)
    regression_means, regression_stds = build_regression_stats(train_frame, regression_targets)

    train_dataset = MoleculeDataset(
        frame=train_frame,
        tokenizer=tokenizer,
        smiles_column=args.smiles_column,
        regression_targets=regression_targets,
        classification_targets=classification_targets,
        regression_means=regression_means,
        regression_stds=regression_stds,
        max_length=args.max_length,
    )
    val_dataset = MoleculeDataset(
        frame=val_frame,
        tokenizer=tokenizer,
        smiles_column=args.smiles_column,
        regression_targets=regression_targets,
        classification_targets=classification_targets,
        regression_means=regression_means,
        regression_stds=regression_stds,
        max_length=args.max_length,
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
    model = MolecularPropertyModel(
        model_name=args.model_name,
        regression_targets=regression_targets,
        classification_targets=classification_targets,
        dropout=args.dropout,
        mlp_hidden_dims=mlp_hidden_dims,
        trust_remote_code=args.trust_remote_code,
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
        print(json.dumps(epoch_record))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_name": args.model_name,
                    "smiles_column": args.smiles_column,
                    "regression_targets": regression_targets,
                    "classification_targets": classification_targets,
                    "regression_means": regression_means,
                    "regression_stds": regression_stds,
                    "max_length": args.max_length,
                    "dropout": args.dropout,
                    "mlp_hidden_dims": mlp_hidden_dims,
                    "trust_remote_code": args.trust_remote_code,
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
                "model_name": args.model_name,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "dropout": args.dropout,
                "mlp_hidden_dims": mlp_hidden_dims,
                "max_length": args.max_length,
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

    test_dataset = MoleculeDataset(
        frame=test_frame,
        tokenizer=tokenizer,
        smiles_column=args.smiles_column,
        regression_targets=regression_targets,
        classification_targets=classification_targets,
        regression_means=regression_means,
        regression_stds=regression_stds,
        max_length=args.max_length,
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


def run_from_cli(
    default_model_name: str,
    default_output_dir: Path,
    default_trust_remote_code: bool,
) -> None:
    args = parse_args(
        default_model_name=default_model_name,
        default_output_dir=default_output_dir,
        default_trust_remote_code=default_trust_remote_code,
    )
    train_and_predict(args)


if __name__ == "__main__":
    run_from_cli(
        default_model_name="seyonec/ChemBERTa-zinc-base-v1",
        default_output_dir=Path("outputs"),
        default_trust_remote_code=False,
    )
