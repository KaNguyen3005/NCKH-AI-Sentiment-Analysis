"""Train the BiLSTM sentiment classifier with the cached FastText vectors.

This trainer uses only the pre-split training and validation CSV files. It
loads the existing compact FastText cache and never rebuilds embeddings,
downloads external data, or reads the test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bilstm import BiLSTM  # noqa: E402
from src.preprocessing import tokenize_for_ml_rnn  # noqa: E402


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1
EMBEDDING_DIM = 300
NUM_CLASSES = 3
NUM_LAYERS = 1
TEXT_COLUMN = "text_clean"
LABEL_COLUMN = "label_id"
LABEL_IDS = (0, 1, 2)
LABEL_MAPPING = {0: "Tiêu cực", 1: "Bình thường", 2: "Tích cực"}
DEFAULT_FASTTEXT_COVERAGE = 0.3712
LOG_COLUMNS = [
    "date",
    "model",
    "config",
    "train_size",
    "val_acc",
    "val_f1_macro",
    "val_f1_negative",
    "val_f1_neutral",
    "val_f1_positive",
    "train_time_min",
    "notes",
]


@dataclass(frozen=True)
class ValidationMetrics:
    """Validation loss and classification metrics for one epoch."""

    loss: float
    accuracy: float
    f1_macro: float
    f1_negative: float
    f1_neutral: float
    f1_positive: float


class RNNSentimentDataset(Dataset[tuple[Tensor, Tensor]]):
    """Pre-tokenized variable-length examples for an RNN classifier."""

    def __init__(
        self,
        frame: pd.DataFrame,
        word2idx: dict[str, int],
        max_length: int,
        split_name: str,
    ) -> None:
        self.sequences: list[Tensor] = []
        self.labels: list[int] = []
        unk_idx = word2idx[UNK_TOKEN]

        rows = zip(frame[TEXT_COLUMN].tolist(), frame[LABEL_COLUMN].tolist())
        for text_value, label in tqdm(
            rows,
            total=len(frame),
            desc=f"Tokenizing {split_name}",
        ):
            text = "" if pd.isna(text_value) else str(text_value)
            tokens = tokenize_for_ml_rnn(text).split()
            token_ids = [word2idx.get(token, unk_idx) for token in tokens[:max_length]]
            if not token_ids:
                token_ids = [unk_idx]

            self.sequences.append(torch.tensor(token_ids, dtype=torch.long))
            self.labels.append(int(label))

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return one variable-length token-ID tensor and its label."""
        return self.sequences[index], torch.tensor(self.labels[index], dtype=torch.long)


def collate_batch(batch: Sequence[tuple[Tensor, Tensor]]) -> dict[str, Tensor]:
    """Pad variable-length sequences and retain their true lengths."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([sequence.numel() for sequence in sequences], dtype=torch.long)
    input_ids = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=PAD_IDX,
    )
    return {
        "input_ids": input_ids,
        "lengths": lengths,
        "labels": torch.stack(labels),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a 3-class BiLSTM sentiment model with cached FastText vectors."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/processed/train.csv"),
        help="Training CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=Path("data/processed/val.csv"),
        help="Validation CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--word2idx-path",
        type=Path,
        default=Path("models/fasttext/word2idx.json"),
        help="Cached word-to-index JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=Path("models/fasttext/embedding_matrix.npy"),
        help="Cached FastText matrix (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/baseline"),
        help="Checkpoint directory (default: %(default)s).",
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
        help="Experiment CSV path (default: %(default)s).",
    )
    parser.add_argument(
        "--run-name",
        default="bilstm_ft_base",
        help="Run identifier used in artifact names (default: %(default)s).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Maximum tokens per example (default: %(default)s).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="LSTM hidden dimension (default: %(default)s).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Classifier dropout probability (default: %(default)s).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size (default: %(default)s).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Maximum number of epochs (default: %(default)s).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early-stopping patience on validation macro-F1 (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device (default: %(default)s).",
    )
    parser.add_argument(
        "--freeze-embeddings",
        action="store_true",
        help="Keep cached FastText embedding weights frozen.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use at most 500 train and 200 validation samples for one epoch.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI values and required local files."""
    required_files = {
        "training CSV": args.train_path,
        "validation CSV": args.val_path,
        "word2idx cache": args.word2idx_path,
        "embedding cache": args.embedding_path,
    }
    for description, path in required_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")

    if not args.run_name or Path(args.run_name).name != args.run_name:
        raise ValueError("--run-name must be a non-empty file-name-safe identifier.")
    if args.max_length < 1:
        raise ValueError("--max-length must be at least 1.")
    if args.hidden_dim < 1:
        raise ValueError("--hidden-dim must be at least 1.")
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


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda into an available PyTorch device."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_fasttext_cache(
    word2idx_path: Path, embedding_path: Path
) -> tuple[dict[str, int], Tensor]:
    """Load and strictly validate the compact FastText cache."""
    with word2idx_path.open("r", encoding="utf-8") as file:
        raw_word2idx = json.load(file)
    if not isinstance(raw_word2idx, dict):
        raise ValueError(f"word2idx cache must be a JSON object: {word2idx_path}")

    word2idx: dict[str, int] = {}
    for token, index in raw_word2idx.items():
        if not isinstance(token, str) or not isinstance(index, int):
            raise ValueError("word2idx must map string tokens to integer indices.")
        word2idx[token] = index

    if word2idx.get(PAD_TOKEN) != PAD_IDX:
        raise ValueError("Invalid word2idx cache: <PAD> must have index 0.")
    if word2idx.get(UNK_TOKEN) != UNK_IDX:
        raise ValueError("Invalid word2idx cache: <UNK> must have index 1.")
    if set(word2idx.values()) != set(range(len(word2idx))):
        raise ValueError("word2idx indices must be unique and contiguous from 0.")

    embeddings = np.load(embedding_path, allow_pickle=False)
    expected_shape = (len(word2idx), EMBEDDING_DIM)
    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Invalid embedding matrix shape: {embeddings.shape}; expected {expected_shape}."
        )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if not np.all(embeddings[PAD_IDX] == 0.0):
        raise ValueError("Invalid embedding cache: the <PAD> vector must be all zeros.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Invalid embedding cache: matrix contains NaN or infinity.")

    return word2idx, torch.from_numpy(embeddings.copy()).to(dtype=torch.float32)


def load_and_validate_frame(path: Path, split_name: str) -> pd.DataFrame:
    """Load a split and validate required columns and three-class labels."""
    frame = pd.read_csv(path, encoding="utf-8")
    missing_columns = [
        column for column in (TEXT_COLUMN, LABEL_COLUMN) if column not in frame.columns
    ]
    if missing_columns:
        raise KeyError(
            f"{split_name} CSV {path} is missing columns: {', '.join(missing_columns)}"
        )
    if frame.empty:
        raise ValueError(f"{split_name} CSV is empty: {path}")

    try:
        numeric_labels = pd.to_numeric(frame[LABEL_COLUMN], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{split_name} column {LABEL_COLUMN!r} must contain integer labels 0, 1, or 2."
        ) from exc
    if numeric_labels.isna().any():
        raise ValueError(f"{split_name} column {LABEL_COLUMN!r} contains missing labels.")

    label_values = numeric_labels.to_numpy(dtype=np.float64)
    if not np.equal(label_values, np.floor(label_values)).all():
        raise ValueError(f"{split_name} labels must be integers in {LABEL_IDS}.")
    invalid_labels = sorted(set(label_values.astype(np.int64)).difference(LABEL_IDS))
    if invalid_labels:
        raise ValueError(
            f"{split_name} contains invalid labels {invalid_labels}; expected only {LABEL_IDS}."
        )

    validated = frame.copy()
    validated[LABEL_COLUMN] = label_values.astype(np.int64)
    return validated


def compute_class_weights(labels: pd.Series) -> Tensor:
    """Compute N / (3 * class_count) using training labels only."""
    counts = np.bincount(labels.to_numpy(dtype=np.int64), minlength=NUM_CLASSES)
    missing_classes = [label for label, count in enumerate(counts) if count == 0]
    if missing_classes:
        raise ValueError(
            f"Cannot compute class weights; train split has no samples for {missing_classes}."
        )
    sample_count = int(counts.sum())
    weights = sample_count / (NUM_CLASSES * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)


def make_data_loader(
    dataset: RNNSentimentDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Create a deterministic, Windows-safe DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=pin_memory,
        generator=generator,
    )


def train_one_epoch(
    model: BiLSTM,
    data_loader: DataLoader[tuple[Tensor, Tensor]],
    loss_fn: nn.CrossEntropyLoss,
    optimizer: Adam,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    """Train for one epoch and return mean sample-weighted loss."""
    model.train()
    total_loss = 0.0
    sample_count = 0

    for batch in tqdm(data_loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        lengths = batch["lengths"]
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, lengths=lengths)
        loss = loss_fn(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size

    return total_loss / sample_count


def evaluate_model(
    model: BiLSTM,
    data_loader: DataLoader[tuple[Tensor, Tensor]],
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
) -> ValidationMetrics:
    """Evaluate loss, accuracy, macro-F1, and per-class F1."""
    model.eval()
    total_loss = 0.0
    sample_count = 0
    true_labels: list[int] = []
    predictions: list[int] = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            lengths = batch["lengths"]
            labels = batch["labels"].to(device, non_blocking=True)

            logits = model(input_ids, lengths=lengths)
            loss = loss_fn(logits, labels)
            predicted = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            sample_count += batch_size
            true_labels.extend(labels.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())

    per_class_f1 = f1_score(
        true_labels,
        predictions,
        labels=list(LABEL_IDS),
        average=None,
        zero_division=0,
    )
    macro_f1 = f1_score(
        true_labels,
        predictions,
        labels=list(LABEL_IDS),
        average="macro",
        zero_division=0,
    )
    return ValidationMetrics(
        loss=total_loss / sample_count,
        accuracy=float(accuracy_score(true_labels, predictions)),
        f1_macro=float(macro_f1),
        f1_negative=float(per_class_f1[0]),
        f1_neutral=float(per_class_f1[1]),
        f1_positive=float(per_class_f1[2]),
    )


def get_log_fieldnames(log_path: Path) -> list[str]:
    """Read and validate an existing log header without modifying it."""
    if not log_path.exists() or log_path.stat().st_size == 0:
        return LOG_COLUMNS.copy()
    with log_path.open("r", encoding="utf-8", newline="") as file:
        fieldnames = next((row for row in csv.reader(file) if row), None)
    if not fieldnames:
        return LOG_COLUMNS.copy()
    missing = [column for column in LOG_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(
            f"Experiment log {log_path} is missing required columns: {', '.join(missing)}"
        )
    return fieldnames


def load_fasttext_coverage(word2idx_path: Path) -> tuple[float, Path]:
    """Read coverage from the cache metadata beside word2idx when available."""
    metadata_path = word2idx_path.parent / "fasttext_cache_meta.json"
    if not metadata_path.is_file():
        return DEFAULT_FASTTEXT_COVERAGE, metadata_path
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    try:
        coverage = float(metadata["coverage"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid FastText coverage in cache metadata: {metadata_path}"
        ) from exc
    if not 0.0 <= coverage <= 1.0:
        raise ValueError(f"FastText coverage must be in [0, 1]: {metadata_path}")
    return coverage, metadata_path


def save_checkpoint(
    path: Path,
    model: BiLSTM,
    optimizer: Adam,
    epoch: int,
    metrics: ValidationMetrics,
    model_config: dict[str, object],
    training_config: dict[str, object],
    word2idx: dict[str, int],
    cache_paths: dict[str, str],
) -> None:
    """Save the best model and all information required to reproduce it."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_f1_macro": metrics.f1_macro,
            "val_metrics": asdict(metrics),
            "model_config": model_config,
            "training_config": training_config,
            "word2idx": word2idx,
            "label_mapping": LABEL_MAPPING,
            "cache_paths": cache_paths,
        },
        path,
    )


def save_learning_curve(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    path: Path,
    run_name: str,
) -> None:
    """Save train and validation loss on one learning-curve figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train loss")
    plt.plot(epochs, val_losses, marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"BiLSTM + FastText learning curve: {run_name}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def append_experiment_log(
    log_path: Path,
    fieldnames: Sequence[str],
    args: argparse.Namespace,
    train_size: int,
    best_epoch: int,
    best_metrics: ValidationMetrics,
    class_weights: Tensor,
    train_time_min: float,
    early_stopped: bool,
    coverage: float,
) -> None:
    """Append exactly one successful non-debug run to the experiment log."""
    class_weight_text = "/".join(f"{weight:.6f}" for weight in class_weights.tolist())
    config = (
        f"run_name={args.run_name}, max_length={args.max_length}, "
        f"hidden_dim={args.hidden_dim}, dropout={args.dropout}, "
        f"lr={args.learning_rate}, batch_size={args.batch_size}, "
        f"epochs={args.epochs}, class_weight=[{class_weight_text}], "
        f"freeze_embeddings={args.freeze_embeddings}, seed={args.seed}"
    )
    note_parts = [f"best epoch={best_epoch}"]
    if early_stopped:
        note_parts.append("early stopping")
    note_parts.append(f"FastText coverage {coverage:.2%}")

    row: dict[str, object] = {field: "" for field in fieldnames}
    row.update(
        {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "model": "BILSTM_FASTTEXT",
            "config": config,
            "train_size": train_size,
            "val_acc": f"{best_metrics.accuracy:.4f}",
            "val_f1_macro": f"{best_metrics.f1_macro:.4f}",
            "val_f1_negative": f"{best_metrics.f1_negative:.4f}",
            "val_f1_neutral": f"{best_metrics.f1_neutral:.4f}",
            "val_f1_positive": f"{best_metrics.f1_positive:.4f}",
            "train_time_min": f"{train_time_min:.4f}",
            "notes": "; ".join(note_parts),
        }
    )
    if "epoch" in row:
        row["epoch"] = best_epoch

    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """Run BiLSTM/FastText training, checkpointing, plotting, and logging."""
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)

    log_fieldnames = None if args.debug else get_log_fieldnames(args.log_path)
    word2idx, pretrained_embeddings = load_fasttext_cache(
        args.word2idx_path, args.embedding_path
    )
    coverage, metadata_path = load_fasttext_coverage(args.word2idx_path)
    train_frame = load_and_validate_frame(args.train_path, "train")
    val_frame = load_and_validate_frame(args.val_path, "validation")
    class_weights = compute_class_weights(train_frame[LABEL_COLUMN])

    if args.debug:
        train_frame = train_frame.sample(
            n=min(500, len(train_frame)), random_state=args.seed
        ).reset_index(drop=True)
        val_frame = val_frame.sample(
            n=min(200, len(val_frame)), random_state=args.seed
        ).reset_index(drop=True)
        epochs_to_run = 1
        artifact_run_name = f"{args.run_name}_debug"
    else:
        epochs_to_run = args.epochs
        artifact_run_name = args.run_name

    train_dataset = RNNSentimentDataset(
        train_frame, word2idx, args.max_length, "train"
    )
    val_dataset = RNNSentimentDataset(val_frame, word2idx, args.max_length, "validation")
    pin_memory = device.type == "cuda"
    train_loader = make_data_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        seed=args.seed,
        pin_memory=pin_memory,
    )
    val_loader = make_data_loader(
        val_dataset,
        args.batch_size,
        shuffle=False,
        seed=args.seed,
        pin_memory=pin_memory,
    )

    model_config: dict[str, object] = {
        "vocab_size": len(word2idx),
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": args.hidden_dim,
        "num_layers": NUM_LAYERS,
        "bidirectional": True,
        "dropout": args.dropout,
        "num_classes": NUM_CLASSES,
        "padding_idx": PAD_IDX,
        "freeze_embeddings": args.freeze_embeddings,
    }
    model = BiLSTM(
        vocab_size=len(word2idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=NUM_LAYERS,
        bidirectional=True,
        dropout=args.dropout,
        num_classes=NUM_CLASSES,
        padding_idx=PAD_IDX,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=args.freeze_embeddings,
    ).to(device)
    optimizer = Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    max_grad_norm = 5.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"{artifact_run_name}_best.pt"
    figure_path = args.figure_dir / f"{artifact_run_name}_learning_curve.png"
    cache_paths = {
        "word2idx": str(args.word2idx_path),
        "embedding_matrix": str(args.embedding_path),
        "metadata": str(metadata_path),
    }
    training_config: dict[str, object] = {
        "train_path": str(args.train_path),
        "val_path": str(args.val_path),
        "run_name": args.run_name,
        "artifact_run_name": artifact_run_name,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": epochs_to_run,
        "patience": args.patience,
        "max_grad_norm": max_grad_norm,
        "seed": args.seed,
        "device": str(device),
        "class_weights": class_weights.tolist(),
        "debug": args.debug,
    }

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Vocabulary size: {len(word2idx)}")
    print(f"Class weights: {class_weights.tolist()}")

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_f1 = -1.0
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
            device,
            max_grad_norm,
        )
        val_metrics = evaluate_model(model, val_loader, loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_metrics.loss)

        print(
            f"Epoch {epoch}/{epochs_to_run} | "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics.loss:.4f} | "
            f"val_accuracy={val_metrics.accuracy:.4f} | "
            f"val_f1_macro={val_metrics.f1_macro:.4f} | "
            f"f1_label_0={val_metrics.f1_negative:.4f} | "
            f"f1_label_1={val_metrics.f1_neutral:.4f} | "
            f"f1_label_2={val_metrics.f1_positive:.4f}"
        )

        if val_metrics.f1_macro > best_f1:
            best_f1 = val_metrics.f1_macro
            best_epoch = epoch
            best_metrics = val_metrics
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                val_metrics,
                model_config,
                training_config,
                word2idx,
                cache_paths,
            )
            print(f"Saved best checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience and epoch < epochs_to_run:
                early_stopped = True
                print(f"Early stopping after epoch {epoch}.")
                break

    train_time_min = (time.perf_counter() - training_start) / 60.0
    if best_metrics is None:
        raise RuntimeError("Training finished without producing validation metrics.")

    save_learning_curve(train_losses, val_losses, figure_path, artifact_run_name)
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
            class_weights,
            train_time_min,
            early_stopped,
            coverage,
        )
        print(f"Appended experiment log: {args.log_path}")
    else:
        print("Debug run: experiment log was not modified.")

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro-F1: {best_metrics.f1_macro:.4f}")
    print(f"Learning curve: {figure_path}")
    print(f"Training time: {train_time_min:.2f} minutes")


if __name__ == "__main__":
    main()
