from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.models.plm_classifier import PLMClassifier
from src.preprocessing import prepare_for_mbert


REQUIRED_FILES = (
    "model.safetensors",
    "model_config.json",
    "training_meta.json",
    "tokenizer_config.json",
)


def parse_args() -> argparse.Namespace:
    """Parse checkpoint validation options."""
    parser = argparse.ArgumentParser(
        description="Check a saved PLMClassifier checkpoint."
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT
        / "models"
        / "baseline"
        / "mbert_smoke_debug",
        help="Directory containing the saved PLM checkpoint.",
    )

    expectation_group = parser.add_mutually_exclusive_group()

    expectation_group.add_argument(
        "--expect-debug",
        action="store_true",
        help="Require checkpoint metadata debug=true.",
    )

    expectation_group.add_argument(
        "--expect-non-debug",
        action="store_true",
        help="Require checkpoint metadata debug=false.",
    )

    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}.")

    return data


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_dir}"
        )

    for filename in REQUIRED_FILES:
        path = checkpoint_dir / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Required checkpoint file is missing: {path}"
            )

    model_config = read_json(checkpoint_dir / "model_config.json")
    training_meta = read_json(checkpoint_dir / "training_meta.json")

    debug_value = training_meta.get("debug")

    if not isinstance(debug_value, bool):
        raise ValueError(
            "training_meta.json must contain a boolean debug field."
        )

    if args.expect_debug and debug_value is not True:
        raise ValueError(
            "Expected a debug checkpoint, but metadata contains debug=false."
        )

    if args.expect_non_debug and debug_value is not False:
        raise ValueError(
            "Expected a non-debug checkpoint, but metadata contains debug=true."
        )

    if training_meta.get("selection_metric") != "validation_macro_f1":
        raise ValueError(
            "Checkpoint was not selected by validation Macro-F1."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
    )

    model = PLMClassifier(
        model_name=model_config["model_name"],
        num_classes=int(model_config["num_classes"]),
        head_type=model_config["head_type"],
        dropout=float(model_config["dropout"]),
    )

    state_dict = load_file(
        str(checkpoint_dir / "model.safetensors"),
        device="cpu",
    )

    incompatible_keys = model.load_state_dict(
        state_dict,
        strict=True,
    )

    if incompatible_keys.missing_keys:
        raise RuntimeError(
            f"Missing state-dict keys: {incompatible_keys.missing_keys}"
        )

    if incompatible_keys.unexpected_keys:
        raise RuntimeError(
            "Unexpected state-dict keys: "
            f"{incompatible_keys.unexpected_keys}"
        )

    samples = [
        "AI hỗ trợ tôi học tập hiệu quả.",
        "Ứng dụng này không hữu ích và thường trả lời sai.",
    ]

    cleaned_samples = [
        prepare_for_mbert(sample)
        for sample in samples
    ]

    encoding = tokenizer(
        cleaned_samples,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )

    model.eval()

    with torch.no_grad():
        logits = model(**encoding)

    if logits.shape != (2, 3):
        raise RuntimeError(
            f"Expected logits shape (2, 3), got {tuple(logits.shape)}."
        )

    if not torch.isfinite(logits).all():
        raise RuntimeError("Loaded checkpoint produced non-finite logits.")

    probabilities = torch.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1)

    weights_path = checkpoint_dir / "model.safetensors"

    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Tokenizer class: {type(tokenizer).__name__}")
    print(f"Architecture: {model_config['architecture']}")
    print(f"Model name: {model_config['model_name']}")
    print(f"Head type: {model_config['head_type']}")
    print(f"Debug checkpoint: {training_meta['debug']}")
    print(f"Best epoch: {training_meta['best_epoch']}")
    print(
        "Selection metric: "
        f"{training_meta['selection_metric']}"
    )
    print(f"State dict tensors: {len(state_dict)}")
    print(f"Missing keys: {incompatible_keys.missing_keys}")
    print(f"Unexpected keys: {incompatible_keys.unexpected_keys}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Finite logits: {torch.isfinite(logits).all().item()}")
    print(f"Predictions: {predictions.tolist()}")
    print(f"Probabilities shape: {tuple(probabilities.shape)}")
    print(f"Checkpoint size: {weights_path.stat().st_size} bytes")
    print(f"Checkpoint SHA-256: {calculate_sha256(weights_path)}")
    print("PLM checkpoint reload: ALL PASSED")


if __name__ == "__main__":
    main()