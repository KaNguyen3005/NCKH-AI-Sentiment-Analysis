import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Khởi tạo dataset.
        :param texts: Danh sách / mảng các câu văn bản.
        :param labels: Nhãn tương ứng.
        :param tokenizer: Tokenizer (từ Hugging Face transformers).
        :param max_length: Chiều dài tối đa của sequence sau khi tokenize.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize văn bản, thêm các token đặc biệt, padding và cắt bớt (truncation)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_length, batch_size, text_col='text', label_col='label', shuffle=True):
    """
    Hàm tiện ích giúp dễ dàng tạo DataLoader từ Pandas DataFrame.
    """
    dataset = SentimentDataset(
        texts=df[text_col].to_numpy(),
        labels=df[label_col].to_numpy(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
