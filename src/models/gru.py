from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRUClassifier(nn.Module):
    """GRU classifier that produces raw logits for sentiment classes."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
        num_classes: int = 3,
        padding_idx: Optional[int] = None,
        pretrained_embeddings: Optional[Tensor] = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()

        if pretrained_embeddings is not None:
            if pretrained_embeddings.ndim != 2:
                raise ValueError("pretrained_embeddings must have shape [vocab_size, embedding_dim].")
            if pretrained_embeddings.shape != (vocab_size, embedding_dim):
                raise ValueError(
                    "pretrained_embeddings shape must match "
                    "[vocab_size, embedding_dim]."
                )

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        if pretrained_embeddings is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(
                    pretrained_embeddings.to(
                        device=self.embedding.weight.device,
                        dtype=self.embedding.weight.dtype,
                    )
                )
        self.embedding.weight.requires_grad = not freeze_embeddings

        # GRU's built-in dropout is only applied between stacked GRU layers.
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=gru_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.num_directions = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(
        self,
        input_ids: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        """Return class logits for token IDs shaped ``[batch_size, sequence_length]``."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, sequence_length].")

        embedded = self.embedding(input_ids)
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != input_ids.shape[0]:
                raise ValueError("lengths must have shape [batch_size].")
            _, hidden = self.gru(
                pack_padded_sequence(
                    embedded,
                    lengths.detach().to(device="cpu", dtype=torch.long),
                    batch_first=True,
                    enforce_sorted=False,
                )
            )
        else:
            _, hidden = self.gru(embedded)

        if self.num_directions == 2:
            final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            final_hidden = hidden[-1]

        return self.classifier(self.dropout(final_hidden))
