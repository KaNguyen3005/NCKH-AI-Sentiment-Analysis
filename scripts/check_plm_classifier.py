"""Lightweight checks for PLMClassifier without downloading full mBERT."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import BertConfig, BertModel


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.models.plm_classifier import PLMClassifier


def build_tiny_bert() -> BertModel:
    """Create a very small random BERT encoder for local unit checks."""
    config = BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        type_vocab_size=2,
    )
    return BertModel(config)


def check_head(head_type: str) -> None:
    """Check forward, shape, finiteness, and backward for one head."""
    torch.manual_seed(42)

    model = PLMClassifier(
        model_name="unit-test-tiny-bert",
        num_classes=3,
        head_type=head_type,
        dropout=0.1,
        encoder=build_tiny_bert(),
    )

    input_ids = torch.randint(
        low=0,
        high=128,
        size=(4, 12),
        dtype=torch.long,
    )

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )

    token_type_ids = torch.zeros_like(input_ids)

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )

    assert logits.shape == (4, 3)
    assert torch.isfinite(logits).all()

    loss = logits.square().mean()
    loss.backward()

    classifier_gradient = model.classifier.weight.grad

    assert classifier_gradient is not None
    assert torch.isfinite(classifier_gradient).all()

    config = model.get_model_config()

    assert config["architecture"] == "plm_classifier"
    assert config["num_classes"] == 3
    assert config["hidden_size"] == 32

    print(
        f"head={head_type}: "
        f"logits={tuple(logits.shape)}, "
        f"finite=True, backward=OK"
    )


def check_without_token_type_ids() -> None:
    """Confirm models can run when token_type_ids are absent."""
    model = PLMClassifier(
        model_name="unit-test-tiny-bert",
        num_classes=3,
        head_type="cls",
        encoder=build_tiny_bert(),
    )

    input_ids = torch.randint(0, 128, (2, 8))
    attention_mask = torch.ones_like(input_ids)

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()

    print("without token_type_ids: OK")


def check_invalid_head() -> None:
    """Confirm invalid configuration is rejected early."""
    try:
        PLMClassifier(
            model_name="unit-test-tiny-bert",
            head_type="invalid_head",
            encoder=build_tiny_bert(),
        )
    except ValueError as exc:
        assert "Unsupported head_type" in str(exc)
        print("invalid head validation: OK")
        return

    raise AssertionError("Invalid head_type was not rejected.")


def main() -> None:
    check_head("cls")
    check_head("mean_pooling")
    check_without_token_type_ids()
    check_invalid_head()

    print("PLMClassifier checks: ALL PASSED")


if __name__ == "__main__":
    main()