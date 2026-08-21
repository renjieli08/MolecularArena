import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from multitask_transformer_finetune import (
    CSVLogger,
    MoleculeDataset,
    MolecularPropertyModel,
    PROJECT_ROOT,
    evaluate,
    resolve_input_path,
    run_epoch,
    save_predictions,
    set_seed,
    train_val_split,
    collect_predictions,
)


def load_json_if_exists(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_last_global_step(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    frame = pd.read_csv(log_path)
    if "global_step" not in frame.columns or frame.empty:
        return 0
    series = pd.to_numeric(frame["global_step"], errors="coerce").dropna()
    return int(series.max()) if not series.empty else 0


def infer_last_epoch(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    frame = pd.read_csv(log_path)
    if "epoch" not in frame.columns or frame.empty:
        return 0
    series = pd.to_numeric(frame["epoch"], errors="coerce").dropna()
    return int(series.max()) if not series.empty else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue fine-tuning from a saved multitask molecular transformer checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    parser.add_argument("--additional-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--regression-loss-weight", type=float, default=1.0)
    parser.add_argument("--classification-loss-weight", type=float, default=1.0)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--no-test-predictions", action="store_true")
    parser.add_argument("--reset-optimizer", action="store_true")
    return parser.parse_args()


def choose_path(
    explicit_path: Optional[Path],
    config_value: Optional[object],
    default_path: Path,
) -> Path:
    if explicit_path is not None:
        return resolve_input_path(explicit_path)
    if config_value:
        return resolve_input_path(Path(str(config_value)))
    return resolve_input_path(default_path)


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_input_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    output_dir = args.output_dir or checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    training_config = load_json_if_exists(output_dir / "training_config.json")
    log_path = output_dir / "training_log.csv"

    train_csv = choose_path(
        args.train_csv,
        training_config.get("train_csv"),
        PROJECT_ROOT / "Training_and_Eval_Dataset.csv",
    )
    test_csv = choose_path(
        args.test_csv,
        training_config.get("test_csv"),
        PROJECT_ROOT / "Test_dataset.csv",
    )

    seed = args.seed if args.seed is not None else int(training_config.get("seed", 42))
    val_fraction = (
        args.val_fraction if args.val_fraction is not None else float(training_config.get("val_fraction", 0.1))
    )
    batch_size = args.batch_size if args.batch_size is not None else int(training_config.get("batch_size", 16))
    learning_rate = (
        args.learning_rate if args.learning_rate is not None else float(training_config.get("learning_rate", 2e-5))
    )
    weight_decay = (
        args.weight_decay if args.weight_decay is not None else float(training_config.get("weight_decay", 1e-2))
    )

    model_name = checkpoint["model_name"]
    smiles_column = checkpoint.get("smiles_column", training_config.get("smiles_column", "SMILES"))
    regression_targets = checkpoint["regression_targets"]
    classification_targets = checkpoint["classification_targets"]
    regression_means = checkpoint["regression_means"]
    regression_stds = checkpoint["regression_stds"]
    max_length = int(checkpoint.get("max_length", training_config.get("max_length", 128)))
    dropout = float(checkpoint.get("dropout", training_config.get("dropout", 0.1)))
    mlp_hidden_dims = checkpoint.get("mlp_hidden_dims", training_config.get("mlp_hidden_dims", [512, 256]))
    trust_remote_code = bool(checkpoint.get("trust_remote_code", False))

    set_seed(seed)

    frame = pd.read_csv(train_csv)
    if smiles_column not in frame.columns:
        raise ValueError(f"SMILES column '{smiles_column}' was not found in {train_csv}.")
    train_frame, val_frame = train_val_split(frame, val_fraction, seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    train_dataset = MoleculeDataset(
        frame=train_frame,
        tokenizer=tokenizer,
        smiles_column=smiles_column,
        regression_targets=regression_targets,
        classification_targets=classification_targets,
        regression_means=regression_means,
        regression_stds=regression_stds,
        max_length=max_length,
    )
    val_dataset = MoleculeDataset(
        frame=val_frame,
        tokenizer=tokenizer,
        smiles_column=smiles_column,
        regression_targets=regression_targets,
        classification_targets=classification_targets,
        regression_means=regression_means,
        regression_stds=regression_stds,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = MolecularPropertyModel(
        model_name=model_name,
        regression_targets=regression_targets,
        classification_targets=classification_targets,
        dropout=dropout,
        mlp_hidden_dims=mlp_hidden_dims,
        trust_remote_code=trust_remote_code,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    if not args.reset_optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    csv_logger = CSVLogger(
        log_path,
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
        append=True,
    )

    global_step = int(checkpoint.get("global_step", infer_last_global_step(log_path)))
    completed_epoch = int(checkpoint.get("epoch", infer_last_epoch(log_path)))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    start_epoch = completed_epoch + 1
    save_checkpoint = args.save_checkpoint or (output_dir / "resumed_best_model.pt")
    save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    last_checkpoint = output_dir / "resumed_last_model.pt"
    resumed_history = []

    print(f"Resuming from checkpoint: {checkpoint_path}", flush=True)
    print(f"Training CSV: {train_csv}", flush=True)
    print(f"Starting epoch: {start_epoch}", flush=True)
    print(f"Starting global step: {global_step}", flush=True)
    print(f"Saving resumed best checkpoint to: {save_checkpoint}", flush=True)

    for epoch in range(start_epoch, start_epoch + args.additional_epochs):
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
        epoch_record = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        epoch_record.update(metrics)
        resumed_history.append(epoch_record)
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
                    "model_name": model_name,
                    "smiles_column": smiles_column,
                    "regression_targets": regression_targets,
                    "classification_targets": classification_targets,
                    "regression_means": regression_means,
                    "regression_stds": regression_stds,
                    "max_length": max_length,
                    "dropout": dropout,
                    "mlp_hidden_dims": mlp_hidden_dims,
                    "trust_remote_code": trust_remote_code,
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                },
                save_checkpoint,
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_name": model_name,
            "smiles_column": smiles_column,
            "regression_targets": regression_targets,
            "classification_targets": classification_targets,
            "regression_means": regression_means,
            "regression_stds": regression_stds,
            "max_length": max_length,
            "dropout": dropout,
            "mlp_hidden_dims": mlp_hidden_dims,
            "trust_remote_code": trust_remote_code,
            "epoch": start_epoch + args.additional_epochs - 1,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
        },
        last_checkpoint,
    )

    with open(output_dir / "resume_history.json", "w", encoding="utf-8") as handle:
        json.dump(resumed_history, handle, indent=2)

    if args.no_test_predictions:
        return

    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV was not found: {test_csv}")

    test_frame = pd.read_csv(test_csv)
    if smiles_column not in test_frame.columns:
        raise ValueError(f"SMILES column '{smiles_column}' was not found in {test_csv}.")
    test_dataset = MoleculeDataset(
        frame=test_frame,
        tokenizer=tokenizer,
        smiles_column=smiles_column,
        regression_targets=regression_targets,
        classification_targets=classification_targets,
        regression_means=regression_means,
        regression_stds=regression_stds,
        max_length=max_length,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
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
        output_path=output_dir / "resumed_test_predictions.csv",
        predictions=predictions,
        regression_targets=regression_targets,
        classification_targets=classification_targets,
    )


if __name__ == "__main__":
    main()
