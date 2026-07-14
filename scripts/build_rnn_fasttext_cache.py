"""Build a train-only FastText embedding cache for the BiLSTM/GRU models.

The script builds its vocabulary exclusively from the training split, then
streams through a local FastText ``.vec`` file and stores only vectors needed
by that vocabulary. It never reads validation or test data.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import tokenize_for_ml_rnn  # noqa: E402


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build word2idx and a compact FastText embedding matrix for "
            "BiLSTM/GRU using only the training split."
        )
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/processed/train.csv"),
        help="Training CSV used to build the vocabulary (default: %(default)s).",
    )
    parser.add_argument(
        "--text-col",
        default="text_clean",
        help="Training CSV text column (default: %(default)s).",
    )
    parser.add_argument(
        "--fasttext-path",
        type=Path,
        default=Path("models/fasttext/cc.vi.300.vec"),
        help="Local FastText .vec file (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/fasttext"),
        help="Directory for cache outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--max-vocab",
        type=int,
        default=30_000,
        help="Maximum vocabulary size including special tokens (default: %(default)s).",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="Minimum training-token frequency (default: %(default)s).",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=300,
        help="Expected FastText vector dimension (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="NumPy random seed for OOV vectors (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing cache.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate input paths and numeric arguments before doing any work."""
    if not args.train_path.is_file():
        raise FileNotFoundError(f"Training file not found: {args.train_path}")
    if not args.fasttext_path.is_file():
        raise FileNotFoundError(
            f"FastText file not found: {args.fasttext_path}. "
            "Download or place it manually; this script does not download FastText."
        )
    if args.max_vocab < 2:
        raise ValueError("--max-vocab must be at least 2 for <PAD> and <UNK>.")
    if args.min_freq < 1:
        raise ValueError("--min-freq must be at least 1.")
    if args.embedding_dim < 1:
        raise ValueError("--embedding-dim must be at least 1.")


def ensure_outputs_available(output_paths: Iterable[Path], overwrite: bool) -> None:
    """Prevent accidental replacement of an existing cache."""
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Cache output already exists: {formatted}. "
            "Run again with --overwrite to replace it."
        )


def build_vocabulary(
    texts: Iterable[object], max_vocab: int, min_freq: int
) -> tuple[dict[str, int], int]:
    """Build a deterministic token vocabulary and count usable training texts."""
    token_counts: Counter[str] = Counter()
    valid_texts = 0

    for value in texts:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue

        tokens = tokenize_for_ml_rnn(text).split()
        if not tokens:
            continue
        token_counts.update(tokens)
        valid_texts += 1

    eligible_tokens = (
        token
        for token, frequency in token_counts.items()
        if frequency >= min_freq and token not in {PAD_TOKEN, UNK_TOKEN}
    )
    selected_tokens = sorted(
        eligible_tokens,
        key=lambda token: (-token_counts[token], token),
    )[: max_vocab - 2]

    word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    word2idx.update(
        {token: index for index, token in enumerate(selected_tokens, start=2)}
    )
    return word2idx, valid_texts


def _parse_header(line: str) -> tuple[int, int] | None:
    """Return ``(token_count, dimension)`` when a line is a .vec header."""
    parts = line.split()
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def load_selected_fasttext_vectors(
    fasttext_path: Path,
    word2idx: dict[str, int],
    embedding_matrix: np.ndarray,
    embedding_dim: int,
) -> int:
    """Stream a FastText file and copy vectors for vocabulary tokens only."""
    target_tokens = set(word2idx).difference({PAD_TOKEN, UNK_TOKEN})
    found_tokens: set[str] = set()

    def process_vector_line(line: str, line_number: int) -> None:
        parts = line.rstrip().split()
        if not parts:
            return
        if len(parts) != embedding_dim + 1:
            raise ValueError(
                f"Invalid FastText vector dimension at line {line_number}: "
                f"expected {embedding_dim}, got {len(parts) - 1}."
            )

        token = parts[0]
        if token not in target_tokens or token in found_tokens:
            return
        try:
            vector = np.asarray(parts[1:], dtype=np.float32)
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric FastText vector at line {line_number} for token {token!r}."
            ) from exc
        embedding_matrix[word2idx[token]] = vector
        found_tokens.add(token)

    with fasttext_path.open("r", encoding="utf-8", errors="ignore") as file:
        first_line = file.readline()
        if not first_line:
            raise ValueError(f"FastText file is empty: {fasttext_path}")

        header = _parse_header(first_line)
        if header is not None:
            _, header_dim = header
            if header_dim != embedding_dim:
                raise ValueError(
                    "FastText header dimension does not match --embedding-dim: "
                    f"expected {embedding_dim}, got {header_dim}."
                )
            first_vector_line = 2
        else:
            process_vector_line(first_line, 1)
            first_vector_line = 2

        for line_number, line in enumerate(file, start=first_vector_line):
            process_vector_line(line, line_number)
            if len(found_tokens) == len(target_tokens):
                break

    return len(found_tokens)


def verify_cache(
    word2idx_path: Path, embedding_path: Path, embedding_dim: int
) -> None:
    """Reload and validate the two main cache artifacts."""
    with word2idx_path.open("r", encoding="utf-8") as file:
        loaded_word2idx = json.load(file)
    loaded_embeddings = np.load(embedding_path, allow_pickle=False)

    if loaded_word2idx.get(PAD_TOKEN) != PAD_IDX:
        raise ValueError("Cache verification failed: <PAD> index is not 0.")
    if loaded_word2idx.get(UNK_TOKEN) != UNK_IDX:
        raise ValueError("Cache verification failed: <UNK> index is not 1.")
    expected_shape = (len(loaded_word2idx), embedding_dim)
    if loaded_embeddings.shape != expected_shape:
        raise ValueError(
            "Cache verification failed: embedding matrix shape is "
            f"{loaded_embeddings.shape}, expected {expected_shape}."
        )


def main() -> None:
    """Build, save, and verify the train-only FastText cache."""
    args = parse_args()
    validate_args(args)

    word2idx_path = args.output_dir / "word2idx.json"
    embedding_path = args.output_dir / "embedding_matrix.npy"
    metadata_path = args.output_dir / "fasttext_cache_meta.json"
    output_paths = (word2idx_path, embedding_path, metadata_path)
    ensure_outputs_available(output_paths, args.overwrite)

    train_frame = pd.read_csv(args.train_path, encoding="utf-8")
    if args.text_col not in train_frame.columns:
        available = ", ".join(map(str, train_frame.columns))
        raise KeyError(
            f"Text column {args.text_col!r} not found in {args.train_path}. "
            f"Available columns: {available}"
        )

    word2idx, valid_texts = build_vocabulary(
        train_frame[args.text_col], args.max_vocab, args.min_freq
    )
    rng = np.random.default_rng(args.seed)
    embedding_matrix = rng.normal(
        loc=0.0,
        scale=0.05,
        size=(len(word2idx), args.embedding_dim),
    ).astype(np.float32)
    embedding_matrix[PAD_IDX] = 0.0

    fasttext_hits = load_selected_fasttext_vectors(
        args.fasttext_path,
        word2idx,
        embedding_matrix,
        args.embedding_dim,
    )
    vocabulary_tokens = max(len(word2idx) - 2, 0)
    oov_count = vocabulary_tokens - fasttext_hits
    coverage = fasttext_hits / vocabulary_tokens if vocabulary_tokens else 0.0

    metadata = {
        "train_path": str(args.train_path),
        "text_col": args.text_col,
        "fasttext_path": str(args.fasttext_path),
        "vocab_size": len(word2idx),
        "embedding_dim": args.embedding_dim,
        "max_vocab": args.max_vocab,
        "min_freq": args.min_freq,
        "fasttext_hits": fasttext_hits,
        "oov_count": oov_count,
        "coverage": coverage,
        "pad_idx": PAD_IDX,
        "unk_idx": UNK_IDX,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with word2idx_path.open("w", encoding="utf-8") as file:
        json.dump(word2idx, file, ensure_ascii=False, indent=2)
    np.save(embedding_path, embedding_matrix, allow_pickle=False)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    verify_cache(word2idx_path, embedding_path, args.embedding_dim)

    print(f"Train rows read: {len(train_frame)}")
    print(f"Valid texts: {valid_texts}")
    print(f"Vocab size: {len(word2idx)}")
    print(f"FastText hits: {fasttext_hits}")
    print(f"OOV count: {oov_count}")
    print(f"Coverage: {coverage:.2%}")
    print(f"word2idx: {word2idx_path}")
    print(f"embedding matrix: {embedding_path}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
