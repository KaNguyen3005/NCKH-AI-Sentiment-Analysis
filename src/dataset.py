from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

TextPreprocessor = Callable[[str], str]

class SentimentDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer: Any, max_length: int = 128) -> None:
        """
        Khởi tạo dataset.
        :param texts: Danh sách / mảng các câu văn bản.
        :param labels: Nhãn tương ứng.
        :param tokenizer: Tokenizer (từ Hugging Face transformers).
        :param max_length: Chiều dài tối đa của sequence sau khi tokenize.
        """
        if len(texts) != len(labels):
            raise ValueError(
                "texts and labels must have the same length: "
                f"{len(texts)} != {len(labels)}"
            )
        if len(texts) == 0:
            raise ValueError("Dataset must contain at least one sample.")
        if max_length <= 0:
            raise ValueError(
                f"max_length must be greater than 0, got {max_length}."
            )
        # self.texts = texts
        self.texts = [str(text) for text in texts]
        # self.labels = labels
        self.labels = [int(label) for label in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx:int) -> dict[str, torch.Tensor]:
        # text = str(self.texts[idx])
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize văn bản, thêm các token đặc biệt, padding và cắt bớt (truncation)
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            # return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # return {
        #     'input_ids': encoding['input_ids'].flatten(),
        #     'attention_mask': encoding['attention_mask'].flatten(),
        #     'labels': torch.tensor(label, dtype=torch.long)
        # }

        # Keep every tensor produced by the tokenizer.
        # For mBERT, this includes token_type_ids.
        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

def create_data_loader(df: pd.DataFrame,
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    text_col: str = "text_plm",
    label_col: str = "label_id",
    shuffle: bool = True,
    text_preprocessor: TextPreprocessor | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """
    Hàm tiện ích giúp dễ dàng tạo DataLoader từ Pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"df must be a pandas DataFrame, got {type(df).__name__}."
        )

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if max_length <= 0:
        raise ValueError(
            f"max_length must be greater than 0, got {max_length}."
        )

    if batch_size <= 0:
        raise ValueError(
            f"batch_size must be greater than 0, got {batch_size}."
        )

    if num_workers < 0:
        raise ValueError(
            f"num_workers must be non-negative, got {num_workers}."
        )

    required_columns = {text_col, label_col}
    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        raise ValueError(
            "Missing required DataFrame columns: "
            f"{missing_columns}. Available columns: {df.columns.tolist()}"
        )

    if df[text_col].isna().any():
        missing_count = int(df[text_col].isna().sum())
        raise ValueError(
            f"Column {text_col!r} contains {missing_count} missing texts."
        )

    texts = df[text_col].astype(str)

    if text_preprocessor is not None:
        texts = texts.map(text_preprocessor)

    empty_text_mask = texts.str.strip().eq("")

    if empty_text_mask.any():
        empty_count = int(empty_text_mask.sum())
        raise ValueError(
            f"Column {text_col!r} contains {empty_count} empty texts "
            "after preprocessing."
        )

    labels_numeric = pd.to_numeric(df[label_col], errors="raise")

    if labels_numeric.isna().any():
        missing_count = int(labels_numeric.isna().sum())
        raise ValueError(
            f"Column {label_col!r} contains {missing_count} missing labels."
        )

    integer_mask = labels_numeric.map(
        lambda value: float(value).is_integer()
    )

    if not integer_mask.all():
        invalid_values = sorted(
            labels_numeric.loc[~integer_mask].unique().tolist()
        )
        raise ValueError(
            f"Labels must be integers, found: {invalid_values}"
        )

    labels = labels_numeric.astype(int)

    valid_labels = {0, 1, 2}
    invalid_labels = sorted(set(labels.tolist()) - valid_labels)

    if invalid_labels:
        raise ValueError(
            "Labels must belong to {0, 1, 2}, found: "
            f"{invalid_labels}"
        )

    dataset = SentimentDataset(
        texts=texts.tolist(),
        labels=labels.tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
    )
