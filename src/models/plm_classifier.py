"""Shared PLM classifier for mBERT and PhoBERT sentiment baselines."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel


class PLMClassifier(nn.Module):
    """
    Sentiment classifier built on a Hugging Face pretrained encoder.

    Supported classification heads:
    - ``cls``: use the first token representation.
    - ``mean`` / ``mean_pooling``: mean-pool non-padding tokens.

    The model returns raw logits with shape ``[batch_size, num_classes]``.
    Loss computation is intentionally handled by the training script.
    """

    _HEAD_ALIASES = {
        "cls": "cls",
        "mean": "mean",
        "mean_pooling": "mean",
    }

    def __init__(
        self,
        model_name: str,
        num_classes: int = 3,
        head_type: str = "cls",
        dropout: float = 0.1,
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

        if num_classes < 2:
            raise ValueError(
                f"num_classes must be at least 2, got {num_classes}."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout must satisfy 0 <= dropout < 1, got {dropout}."
            )

        normalized_head_type = head_type.strip().lower()

        if normalized_head_type not in self._HEAD_ALIASES:
            supported = sorted(self._HEAD_ALIASES)
            raise ValueError(
                f"Unsupported head_type={head_type!r}. "
                f"Supported values: {supported}"
            )

        self.model_name = model_name
        self.num_classes = num_classes
        self.head_type = self._HEAD_ALIASES[normalized_head_type]
        self.dropout_probability = float(dropout)

        # Dependency injection through ``encoder`` allows lightweight unit
        # tests without downloading the full pretrained model.
        self.encoder = (
            encoder
            if encoder is not None
            else AutoModel.from_pretrained(model_name)
        )

        hidden_size = getattr(self.encoder.config, "hidden_size", None)

        if hidden_size is None:
            raise ValueError(
                "The encoder config does not define hidden_size."
            )

        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(
                f"Invalid encoder hidden_size: {hidden_size!r}"
            )

        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(self.dropout_probability)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def _validate_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None,
    ) -> None:
        """Validate the main encoder input tensors."""
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch_size, sequence_length], "
                f"got {tuple(input_ids.shape)}."
            )

        if attention_mask.ndim != 2:
            raise ValueError(
                "attention_mask must have shape "
                "[batch_size, sequence_length], "
                f"got {tuple(attention_mask.shape)}."
            )

        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                "input_ids and attention_mask must have the same shape: "
                f"{tuple(input_ids.shape)} != "
                f"{tuple(attention_mask.shape)}."
            )

        if token_type_ids is not None:
            if token_type_ids.ndim != 2:
                raise ValueError(
                    "token_type_ids must have shape "
                    "[batch_size, sequence_length], "
                    f"got {tuple(token_type_ids.shape)}."
                )

            if token_type_ids.shape != input_ids.shape:
                raise ValueError(
                    "token_type_ids and input_ids must have the same shape: "
                    f"{tuple(token_type_ids.shape)} != "
                    f"{tuple(input_ids.shape)}."
                )

        valid_token_counts = attention_mask.sum(dim=1)

        if torch.any(valid_token_counts <= 0):
            raise ValueError(
                "Every sample must contain at least one non-padding token."
            )

    @staticmethod
    def _masked_mean_pool(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean-pool representations of tokens selected by attention_mask.

        Padding positions have attention_mask=0 and therefore do not
        contribute to the pooled representation.
        """
        expanded_mask = attention_mask.unsqueeze(-1).to(
            dtype=last_hidden_state.dtype
        )

        summed_embeddings = torch.sum(
            last_hidden_state * expanded_mask,
            dim=1,
        )

        valid_token_counts = expanded_mask.sum(dim=1).clamp_min(1.0)

        return summed_embeddings / valid_token_counts

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        **encoder_kwargs: Any,
    ) -> torch.Tensor:
        """
        Run the pretrained encoder and return raw classification logits.

        Additional supported encoder inputs, such as position_ids, may be
        supplied through encoder_kwargs.
        """
        self._validate_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        if "labels" in encoder_kwargs:
            raise TypeError(
                "PLMClassifier does not compute loss internally. "
                "Remove labels before calling the model."
            )

        encoder_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        encoder_inputs.update(encoder_kwargs)
        encoder_inputs.setdefault("return_dict", True)

        outputs = self.encoder(**encoder_inputs)
        last_hidden_state = outputs.last_hidden_state

        if last_hidden_state.ndim != 3:
            raise RuntimeError(
                "Expected last_hidden_state with shape "
                "[batch_size, sequence_length, hidden_size], "
                f"got {tuple(last_hidden_state.shape)}."
            )

        if self.head_type == "cls":
            features = last_hidden_state[:, 0, :]
        else:
            features = self._masked_mean_pool(
                last_hidden_state=last_hidden_state,
                attention_mask=attention_mask,
            )

        features = self.dropout(features)
        logits = self.classifier(features)

        expected_shape = (input_ids.size(0), self.num_classes)

        if logits.shape != expected_shape:
            raise RuntimeError(
                f"Expected logits shape {expected_shape}, "
                f"got {tuple(logits.shape)}."
            )

        return logits

    def get_model_config(self) -> dict[str, Any]:
        """Return serializable model metadata for checkpoints."""
        return {
            "architecture": "plm_classifier",
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "head_type": self.head_type,
            "dropout": self.dropout_probability,
            "hidden_size": self.hidden_size,
            "encoder_model_type": getattr(
                self.encoder.config,
                "model_type",
                None,
            ),
        }