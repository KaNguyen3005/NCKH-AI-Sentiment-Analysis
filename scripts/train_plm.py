"""Train a shared mBERT/PhoBERT classifier on train and validation only."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import transformers
import yaml
from safetensors.torch import save_file
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import create_data_loader  # noqa: E402
from src.models.plm_classifier import PLMClassifier  # noqa: E402
from src.preprocessing import prepare_for_mbert  # noqa: E402


NUM_CLASSES = 3
LABEL_COLUMN = "label_id"
LABEL_IDS = [0, 1, 2]
LABEL_MAPPING = {0: "Tiêu cực", 1: "Bình thường", 2: "Tích cực"}
MODEL_LOG_NAMES = {
    "mbert": "MBERT_FINETUNED",
    "phobert": "PHOBERT_FINETUNED",
}
REQUIRED_LOG_COLUMNS = [
    "date",
    "model",
    "config",
    "train_size",
    "epoch",
    "val_acc",
    "val_f1_macro",
    "val_f1_negative",
    "val_f1_neutral",
    "val_f1_positive",
    "train_time_min",
    "notes",
]
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class ValidationMetrics:
    """Validation loss and three-class classification metrics."""

    loss: float
    accuracy: float
    f1_macro: float
    f1_negative: float
    f1_neutral: float
    f1_positive: float


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Return a config section after checking that it is a mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section {name!r} must be a mapping.")
    return value


def _repo_path(path: Path) -> Path:
    """Resolve a path relative to the repository root."""
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path) -> Mapping[str, Any]:
    """Load a YAML configuration mapping."""
    resolved_path = _repo_path(path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {resolved_path}")
    with resolved_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return _mapping(config, "root")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI values, using the selected model's YAML section as defaults."""
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config.yaml",
    )
    bootstrap.add_argument(
        "--model-type",
        choices=("mbert", "phobert"),
        default="mbert",
    )
    preliminary, _ = bootstrap.parse_known_args(argv)
    config = load_config(preliminary.config)
    model_config = _mapping(config.get("model"), "model")
    data_config = _mapping(config.get("data"), "data")
    baseline_config = _mapping(config.get("baseline"), "baseline")
    selected_baseline = _mapping(
        baseline_config.get(preliminary.model_type),
        f"baseline.{preliminary.model_type}",
    )

    model_name_key = f"{preliminary.model_type}_name"
    if model_name_key not in model_config:
        raise ValueError(f"Config is missing model.{model_name_key}.")
    for key in ("train_path", "val_path"):
        if key not in data_config:
            raise ValueError(f"Config is missing data.{key}.")

    def baseline_default(key: str, fallback: Any) -> Any:
        return selected_baseline.get(key, fallback)

    default_epochs = baseline_default(
        "epochs",
        baseline_default("num_epochs", 4),
    )
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a shared PLM sentiment classifier using only train and "
            "validation CSV files."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=preliminary.config,
        help="YAML config path (default: repository config.yaml).",
    )
    parser.add_argument(
        "--model-type",
        choices=("mbert", "phobert"),
        default=preliminary.model_type,
        help="Pretrained model family (default: %(default)s).",
    )
    parser.add_argument(
        "--model-name",
        default=model_config[model_name_key],
        help="Hugging Face model/tokenizer name (default from config).",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path(str(data_config["train_path"])),
        help="Training CSV path (default from config).",
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=Path(str(data_config["val_path"])),
        help="Validation CSV path (default from config).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/baseline"),
        help="Parent checkpoint directory (default: %(default)s).",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("data/figures"),
        help="Learning-curve directory (default: %(default)s).",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("results/experiments_log.csv"),
        help="Experiment log path (default: %(default)s).",
    )
    parser.add_argument(
        "--run-name",
        default=f"{preliminary.model_type}_finetuned",
        help="Run directory/artifact identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--head-type",
        choices=("cls", "mean_pooling"),
        default=baseline_default("head_type", "cls"),
    )
    parser.add_argument(
        "--loss-type",
        choices=("ce", "class_weight"),
        default=baseline_default("loss_type", "class_weight"),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=baseline_default("max_length", model_config.get("max_length", 64)),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=baseline_default("dropout", 0.1),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=baseline_default("learning_rate", 2e-5),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=baseline_default("batch_size", 16),
    )
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument(
        "--patience",
        type=int,
        default=baseline_default("patience", 2),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=baseline_default("weight_decay", 0.01),
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=baseline_default("warmup_ratio", 0.1),
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=baseline_default("max_grad_norm", 1.0),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=baseline_default("seed", 42),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-train-size", type=int, default=500)
    parser.add_argument("--debug-val-size", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)
    for attribute in (
        "config",
        "train_path",
        "val_path",
        "output_dir",
        "figure_dir",
        "log_path",
    ):
        setattr(args, attribute, _repo_path(getattr(args, attribute)))
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Validate paths, run identity, and numeric hyperparameters."""
    for split_name, path in (
        ("training", args.train_path),
        ("validation", args.val_path),
    ):
        if path.name.lower() == "test.csv":
            raise ValueError(
                f"Refusing to use locked test.csv as the {split_name} split: {path}"
            )
        if not path.is_file():
            raise FileNotFoundError(f"{split_name.title()} CSV does not exist: {path}")

    if not isinstance(args.model_name, str) or not args.model_name.strip():
        raise ValueError("--model-name must be a non-empty string.")
    if not RUN_NAME_PATTERN.fullmatch(args.run_name):
        raise ValueError(
            "--run-name must start with an alphanumeric character and contain "
            "only letters, numbers, dots, underscores, or hyphens."
        )
    if args.max_length < 1:
        raise ValueError("--max-length must be at least 1.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.patience < 1:
        raise ValueError("--patience must be at least 1.")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative.")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1].")
    if args.max_grad_norm <= 0.0:
        raise ValueError("--max-grad-norm must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.debug_train_size < 1 or args.debug_val_size < 1:
        raise ValueError("Debug sample sizes must be at least 1.")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and CUDA deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    """Resolve the requested device and fail clearly for unavailable CUDA."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_and_validate_frame(
    path: Path,
    split_name: str,
    text_column: str,
) -> pd.DataFrame:
    """Read one CSV and validate text and three-class label columns."""
    frame = pd.read_csv(path, encoding="utf-8")
    if frame.empty:
        raise ValueError(f"{split_name} CSV is empty: {path}")

    missing_columns = [
        column
        for column in (text_column, LABEL_COLUMN)
        if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{split_name} CSV {path} is missing required columns: "
            f"{missing_columns}"
        )
    if frame[text_column].isna().any():
        missing_count = int(frame[text_column].isna().sum())
        raise ValueError(
            f"{split_name} column {text_column!r} contains "
            f"{missing_count} missing texts."
        )

    try:
        numeric_labels = pd.to_numeric(frame[LABEL_COLUMN], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{split_name} column {LABEL_COLUMN!r} must contain integer labels."
        ) from exc
    if numeric_labels.isna().any():
        raise ValueError(
            f"{split_name} column {LABEL_COLUMN!r} contains missing labels."
        )
    label_values = numeric_labels.to_numpy(dtype=np.float64)
    if not np.isfinite(label_values).all():
        raise ValueError(f"{split_name} labels must be finite integers.")
    if not np.equal(label_values, np.floor(label_values)).all():
        raise ValueError(f"{split_name} labels must be integers.")
    invalid_labels = sorted(
        set(label_values.astype(np.int64).tolist()).difference(LABEL_IDS)
    )
    if invalid_labels:
        raise ValueError(
            f"{split_name} labels must belong to {{0, 1, 2}}; "
            f"found {invalid_labels}."
        )
    return frame


def calculate_class_weights(labels: pd.Series) -> Tensor:
    """Compute balanced weights for all three classes from the full train split."""
    numeric_labels = pd.to_numeric(labels, errors="raise").to_numpy(dtype=np.int64)
    missing_classes = sorted(set(LABEL_IDS).difference(numeric_labels.tolist()))
    if missing_classes:
        raise ValueError(
            "Cannot compute balanced class weights because the training split "
            f"has no samples for labels {missing_classes}."
        )
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.asarray(LABEL_IDS, dtype=np.int64),
        y=numeric_labels,
    )
    return torch.tensor(weights, dtype=torch.float32)


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> AdamW:
    """Build AdamW groups with no decay for bias and LayerNorm weights."""
    decay_parameters: list[nn.Parameter] = []
    no_decay_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        normalized_name = name.lower()
        if normalized_name.endswith(".bias") or normalized_name.endswith(
            ("layernorm.weight", "layer_norm.weight")
        ):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    return AdamW(
        [
            {"params": decay_parameters, "weight_decay": weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader[Any],
    loss_fn: nn.Module,
    optimizer: AdamW,
    scheduler: Any,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    """Train one epoch and return sample-weighted mean loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in data_loader:
        labels = batch["labels"].to(device)
        model_inputs = {
            name: tensor.to(device)
            for name, tensor in batch.items()
            if name != "labels"
        }
        optimizer.zero_grad(set_to_none=True)
        logits = model(**model_inputs)
        loss = loss_fn(logits, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Training DataLoader produced no samples.")
    return total_loss / total_samples


def validate_model(
    model: nn.Module,
    data_loader: DataLoader[Any],
    loss_fn: nn.Module,
    device: torch.device,
) -> ValidationMetrics:
    """Calculate sample-weighted validation loss and three-class metrics."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions: list[int] = []
    targets: list[int] = []

    with torch.no_grad():
        for batch in data_loader:
            labels = batch["labels"].to(device)
            model_inputs = {
                name: tensor.to(device)
                for name, tensor in batch.items()
                if name != "labels"
            }
            logits = model(**model_inputs)
            loss = loss_fn(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())

    if total_samples == 0:
        raise RuntimeError("Validation DataLoader produced no samples.")
    per_class_f1 = f1_score(
        targets,
        predictions,
        labels=LABEL_IDS,
        average=None,
        zero_division=0,
    )
    macro_f1 = f1_score(
        targets,
        predictions,
        labels=LABEL_IDS,
        average="macro",
        zero_division=0,
    )
    return ValidationMetrics(
        loss=total_loss / total_samples,
        accuracy=float(accuracy_score(targets, predictions)),
        f1_macro=float(macro_f1),
        f1_negative=float(per_class_f1[0]),
        f1_neutral=float(per_class_f1[1]),
        f1_positive=float(per_class_f1[2]),
    )


def prepare_run_directory(
    output_dir: Path,
    artifact_run_name: str,
    overwrite: bool,
) -> Path:
    """Create an isolated run directory, optionally replacing only that run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / artifact_run_name
    if run_dir.resolve().parent != output_dir.resolve():
        raise ValueError("Resolved run directory escapes --output-dir.")
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Run directory already exists; use --overwrite to replace it: {run_dir}"
            )
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise ValueError(f"Refusing to overwrite a non-directory run path: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=False)
    return run_dir


def save_model_weights(model: nn.Module, run_dir: Path) -> Path:
    """Save a CPU, contiguous copy of the complete best model state dict."""
    state_dict = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    weights_path = run_dir / "model.safetensors"
    save_file(state_dict, str(weights_path))
    return weights_path


def save_checkpoint_metadata(
    run_dir: Path,
    tokenizer: Any,
    model: PLMClassifier,
    training_meta: Mapping[str, Any],
) -> None:
    """Save tokenizer, model configuration, and reproducibility metadata."""
    tokenizer.save_pretrained(run_dir)
    with (run_dir / "model_config.json").open("w", encoding="utf-8") as file:
        json.dump(model.get_model_config(), file, ensure_ascii=False, indent=2)
    with (run_dir / "training_meta.json").open("w", encoding="utf-8") as file:
        json.dump(training_meta, file, ensure_ascii=False, indent=2)


def save_learning_curve(
    train_losses: Sequence[float],
    validation_losses: Sequence[float],
    path: Path,
    artifact_run_name: str,
) -> None:
    """Save train and validation loss curves using a non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train loss")
    plt.plot(epochs, validation_losses, marker="o", label="Validation loss")
    plt.title(f"PLM learning curve: {artifact_run_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def get_log_fieldnames(log_path: Path) -> list[str]:
    """Read the existing CSV header and validate all required columns."""
    if not log_path.is_file():
        raise FileNotFoundError(f"Experiment log does not exist: {log_path}")
    with log_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        fieldnames = next((row for row in reader if row), None)
    if not fieldnames:
        raise ValueError(f"Experiment log has no header: {log_path}")
    missing_columns = [
        column for column in REQUIRED_LOG_COLUMNS if column not in fieldnames
    ]
    if missing_columns:
        raise ValueError(
            f"Experiment log {log_path} is missing required columns: "
            f"{missing_columns}"
        )
    return list(fieldnames)


def append_experiment_log(
    log_path: Path,
    fieldnames: Sequence[str],
    args: argparse.Namespace,
    train_size: int,
    best_epoch: int,
    best_metrics: ValidationMetrics,
    train_time_min: float,
    early_stopped: bool,
    run_dir: Path,
    hyperparameters: Mapping[str, Any],
) -> None:
    """Append exactly one completed non-debug experiment using the existing header."""
    notes = [
        f"best epoch={best_epoch}",
        "selected by validation macro-F1",
    ]
    if early_stopped:
        notes.append("early stopping")
    notes.append(f"checkpoint path={run_dir}")

    row: dict[str, Any] = {column: "" for column in fieldnames}
    row.update(
        {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "model": MODEL_LOG_NAMES[args.model_type],
            "config": json.dumps(
                hyperparameters,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "train_size": train_size,
            "epoch": best_epoch,
            "val_acc": f"{best_metrics.accuracy:.4f}",
            "val_f1_macro": f"{best_metrics.f1_macro:.4f}",
            "val_f1_negative": f"{best_metrics.f1_negative:.4f}",
            "val_f1_neutral": f"{best_metrics.f1_neutral:.4f}",
            "val_f1_positive": f"{best_metrics.f1_positive:.4f}",
            "train_time_min": f"{train_time_min:.4f}",
            "notes": "; ".join(notes),
        }
    )
    with log_path.open("a", encoding="utf-8", newline="") as file:
        csv.DictWriter(file, fieldnames=list(fieldnames)).writerow(row)


def build_hyperparameters(
    args: argparse.Namespace,
    epochs_to_run: int,
    device: torch.device,
) -> dict[str, Any]:
    """Collect all effective training hyperparameters for artifacts and logs."""
    return {
        "head_type": args.head_type,
        "loss_type": args.loss_type,
        "max_length": args.max_length,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "epochs_to_run": epochs_to_run,
        "patience": args.patience,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "requested_device": args.device,
        "resolved_device": str(device),
        "num_workers": args.num_workers,
        "debug_train_size": args.debug_train_size,
        "debug_val_size": args.debug_val_size,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Fine-tune mBERT or PhoBERT and select the best validation macro-F1."""
    args = parse_args(argv)
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)

    text_column = "text" if args.model_type == "mbert" else "text_plm"
    text_preprocessor = prepare_for_mbert if args.model_type == "mbert" else None
    artifact_run_name = (
        f"{args.run_name}_debug" if args.debug else args.run_name
    )
    epochs_to_run = 1 if args.debug else args.epochs
    log_fieldnames = None if args.debug else get_log_fieldnames(args.log_path)
    run_dir = prepare_run_directory(
        args.output_dir,
        artifact_run_name,
        args.overwrite,
    )

    full_train_frame = load_and_validate_frame(
        args.train_path,
        "Training",
        text_column,
    )
    validation_frame = load_and_validate_frame(
        args.val_path,
        "Validation",
        text_column,
    )
    class_weights = calculate_class_weights(full_train_frame[LABEL_COLUMN])

    train_frame = full_train_frame
    if args.debug:
        train_frame = full_train_frame.sample(
            n=min(args.debug_train_size, len(full_train_frame)),
            random_state=args.seed,
        )
        validation_frame = validation_frame.sample(
            n=min(args.debug_val_size, len(validation_frame)),
            random_state=args.seed,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    pin_memory = device.type == "cuda"
    train_loader = create_data_loader(
        train_frame,
        tokenizer,
        args.max_length,
        args.batch_size,
        text_col=text_column,
        label_col=LABEL_COLUMN,
        shuffle=True,
        text_preprocessor=text_preprocessor,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    validation_loader = create_data_loader(
        validation_frame,
        tokenizer,
        args.max_length,
        args.batch_size,
        text_col=text_column,
        label_col=LABEL_COLUMN,
        shuffle=False,
        text_preprocessor=text_preprocessor,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=None,
    )

    model = PLMClassifier(
        model_name=args.model_name,
        num_classes=NUM_CLASSES,
        head_type=args.head_type,
        dropout=args.dropout,
    ).to(device)
    loss_fn: nn.Module
    if args.loss_type == "class_weight":
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_training_steps = len(train_loader) * epochs_to_run
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model type: {args.model_type}")
    print(f"Model name: {args.model_name}")
    print(f"Device: {device}")
    print(f"Train size: {len(train_frame)}")
    print(f"Validation size: {len(validation_frame)}")
    print(f"Text column: {text_column}")
    print(f"Total parameters: {total_parameters:,}")

    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    best_metrics: ValidationMetrics | None = None
    epochs_without_improvement = 0
    early_stopped = False
    training_start = time.perf_counter()

    for epoch in range(1, epochs_to_run + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            scheduler,
            device,
            args.max_grad_norm,
        )
        validation_metrics = validate_model(
            model,
            validation_loader,
            loss_fn,
            device,
        )
        train_losses.append(train_loss)
        validation_losses.append(validation_metrics.loss)

        print(
            f"Epoch {epoch}/{epochs_to_run} | train_loss={train_loss:.4f} | "
            f"val_loss={validation_metrics.loss:.4f} | "
            f"val_accuracy={validation_metrics.accuracy:.4f} | "
            f"val_macro_f1={validation_metrics.f1_macro:.4f} | "
            f"f1_negative={validation_metrics.f1_negative:.4f} | "
            f"f1_neutral={validation_metrics.f1_neutral:.4f} | "
            f"f1_positive={validation_metrics.f1_positive:.4f}"
        )

        if validation_metrics.f1_macro > best_macro_f1:
            best_macro_f1 = validation_metrics.f1_macro
            best_epoch = epoch
            best_metrics = validation_metrics
            epochs_without_improvement = 0
            save_model_weights(model, run_dir)
            print(f"Saved improved checkpoint: {run_dir}")
        else:
            epochs_without_improvement += 1
            if (
                not args.debug
                and epochs_without_improvement >= args.patience
                and epoch < epochs_to_run
            ):
                early_stopped = True
                print(f"Early stopping after epoch {epoch}.")
                break

    train_time_min = (time.perf_counter() - training_start) / 60.0
    if best_metrics is None:
        raise RuntimeError("Training finished without validation metrics.")

    hyperparameters = build_hyperparameters(args, epochs_to_run, device)
    training_meta = {
        "run_name": args.run_name,
        "artifact_run_name": artifact_run_name,
        "model_type": args.model_type,
        "model_name": args.model_name,
        "debug": args.debug,
        "best_epoch": best_epoch,
        "selection_metric": "validation_macro_f1",
        "validation_metrics": asdict(best_metrics),
        "train_size": len(train_frame),
        "validation_size": len(validation_frame),
        "label_mapping": LABEL_MAPPING,
        "class_weights": class_weights.tolist(),
        "hyperparameters": hyperparameters,
        "train_path": str(args.train_path),
        "validation_path": str(args.val_path),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "saved_at": datetime.now().astimezone().isoformat(),
    }
    save_checkpoint_metadata(run_dir, tokenizer, model, training_meta)

    figure_path = args.figure_dir / f"{artifact_run_name}_learning_curve.png"
    save_learning_curve(
        train_losses,
        validation_losses,
        figure_path,
        artifact_run_name,
    )

    log_updated = False
    if not args.debug:
        if log_fieldnames is None:
            raise RuntimeError("Experiment log header was not initialized.")
        append_experiment_log(
            args.log_path,
            log_fieldnames,
            args,
            len(train_frame),
            best_epoch,
            best_metrics,
            train_time_min,
            early_stopped,
            run_dir,
            hyperparameters,
        )
        log_updated = True

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation Macro-F1: {best_metrics.f1_macro:.4f}")
    print(f"Checkpoint directory: {run_dir}")
    print(f"Learning curve path: {figure_path}")
    print(f"Experiment log updated: {'yes' if log_updated else 'no'}")
    print(f"Training time: {train_time_min:.2f} minutes")


if __name__ == "__main__":
    main()
