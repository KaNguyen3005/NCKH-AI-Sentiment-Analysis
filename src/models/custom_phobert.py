import torch
import torch.nn as nn
from transformers import AutoModel

class CustomPhoBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes, head_type="cls"):
        super(CustomPhoBERTClassifier, self).__init__()
        self.phobert = AutoModel.from_pretrained(model_name)
        self.head_type = head_type
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.phobert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.last_hidden_state has shape (batch_size, seq_len, hidden_size)
        last_hidden_state = outputs.last_hidden_state
        
        if self.head_type == "cls":
            # Taking <s> token (index 0)
            features = last_hidden_state[:, 0, :]
        elif self.head_type == "mean_pooling":
            # Mean pooling over valid tokens (ignoring pad tokens via attention_mask)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            features = sum_embeddings / sum_mask
        else:
            raise ValueError(f"Unknown head_type: {self.head_type}")
            
        features = self.dropout(features)
        logits = self.classifier(features)
        
        # Return a dummy object to mimic AutoModelForSequenceClassification output
        return type('obj', (object,), {'logits' : logits})
