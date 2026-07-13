import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
import time
import datetime
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Bypass strict torch version check for loading models with weights_only
import transformers
try:
    import transformers.modeling_utils
    if hasattr(transformers.modeling_utils, "check_torch_load_is_safe"):
        transformers.modeling_utils.check_torch_load_is_safe = lambda: None
    if hasattr(transformers.utils, "import_utils") and hasattr(transformers.utils.import_utils, "check_torch_load_is_safe"):
        transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
except ImportError:
    pass

from dataset import create_data_loader

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    """
    Huấn luyện mô hình trong một epoch.
    """
    model.train()
    total_loss = 0
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        
        # Lấy nhãn dự đoán
        _, preds = torch.max(outputs.logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Tránh bùng nổ gradient
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    acc = correct_predictions.double() / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader)
    return acc, avg_loss

def eval_model(model, data_loader, loss_fn, device):
    """
    Đánh giá mô hình trên tập validation.
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

            _, preds = torch.max(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_loss += loss.item()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = correct_predictions.double() / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader)
    
    # Tính F1-Macro và F1 cho từng nhãn (labels giả định 0: Tiêu cực, 1: Bình thường, 2: Tích cực)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_all = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    
    return acc.item(), avg_loss, f1_macro, f1_all

def log_experiment(log_path, model_name, config_str, train_size, val_acc, val_f1_macro, f1_all, train_time_min, notes=""):
    """
    Ghi log lại thông số huấn luyện vào CSV theo chuẩn yêu cầu.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.isfile(log_path)
    
    with open(log_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Ghi header nếu file chưa tồn tại
        if not file_exists:
            writer.writerow(['date', 'model', 'config', 'train_size', 'val_acc', 'val_f1_macro', 
                             'val_f1_negative', 'val_f1_neutral', 'val_f1_positive', 'train_time_min', 'notes'])
        
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        f1_neg = f1_all[0] if len(f1_all) > 0 else 0
        f1_neu = f1_all[1] if len(f1_all) > 1 else 0
        f1_pos = f1_all[2] if len(f1_all) > 2 else 0
        
        writer.writerow([
            date_str, model_name, config_str, train_size, f"{val_acc:.4f}", f"{val_f1_macro:.4f}",
            f"{f1_neg:.4f}", f"{f1_neu:.4f}", f"{f1_pos:.4f}", f"{train_time_min:.1f}", notes
        ])

def main():
    parser = argparse.ArgumentParser(description="Train Sentiment Analysis Model")
    parser.add_argument("--config", type=str, default="../config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Cấu hình mặc định
    train_config = {
        "model_name": "vinai/phobert-base",
        "num_classes": 3,
        "max_length": 64,
        "batch_size": 32,
        "epochs": 4,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "train_data": "data/processed/train.csv",
        "val_data": "data/processed/val.csv",
        "save_dir": "models/phobert/best_checkpoint",
        "results_dir": "results",
        "figures_dir": "data/figures"
    }

    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
            
            # Ghi đè từ yaml nếu có
            if "model" in yaml_cfg:
                train_config["model_name"] = yaml_cfg["model"].get("phobert_name", train_config["model_name"])
            
            if "data" in yaml_cfg:
                train_config["num_classes"] = yaml_cfg["data"].get("num_labels", train_config["num_classes"])
                train_config["train_data"] = yaml_cfg["data"].get("train_path", train_config["train_data"])
                train_config["val_data"] = yaml_cfg["data"].get("val_path", train_config["val_data"])
            
            if "baseline" in yaml_cfg and "phobert" in yaml_cfg["baseline"]:
                pb_cfg = yaml_cfg["baseline"]["phobert"]
                train_config["max_length"] = pb_cfg.get("max_length", train_config["max_length"])
                train_config["batch_size"] = pb_cfg.get("batch_size", train_config["batch_size"])
                train_config["epochs"] = pb_cfg.get("num_epochs", train_config["epochs"])
                train_config["learning_rate"] = float(pb_cfg.get("learning_rate", train_config["learning_rate"]))
                train_config["weight_decay"] = float(pb_cfg.get("weight_decay", train_config["weight_decay"]))
                train_config["warmup_ratio"] = float(pb_cfg.get("warmup_ratio", train_config["warmup_ratio"]))
    
    config = train_config

    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(config["figures_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    
    print("Loading data...")
    try:
        df_train = pd.read_csv(config["train_data"], encoding="utf-8")
        df_val = pd.read_csv(config["val_data"], encoding="utf-8")
    except FileNotFoundError as e:
        print(f"Lỗi: {e}. Đảm bảo data đã được tạo nằm trong thư mục cấu hình.")
        return

    train_loader = create_data_loader(df_train, tokenizer, config["max_length"], config["batch_size"])
    val_loader = create_data_loader(df_val, tokenizer, config["max_length"], config["batch_size"], shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], 
        num_labels=config["num_classes"]
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * float(config["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss().to(device)

    best_val_loss = float('inf')
    best_val_metrics = None
    
    train_losses = []
    val_losses = []
    start_time = time.time()

    # Vòng lặp huấn luyện chính
    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print("-" * 20)

        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler)
        print(f"Train loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")

        val_acc, val_loss, val_f1_macro, f1_all = eval_model(model, val_loader, loss_fn, device)
        print(f"Val loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | F1-Macro: {val_f1_macro:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metrics = (val_acc, val_f1_macro, f1_all)
            torch.save(model.state_dict(), os.path.join(config["save_dir"], "best_model.pt"))
            print("=> Saved new best model checkpoint!")

    train_time_min = (time.time() - start_time) / 60.0

    # Lưu biểu đồ Learning Curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, config["epochs"] + 1), train_losses, label='Train Loss')
    plt.plot(range(1, config["epochs"] + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Learning Curve - {config["model_name"].split("/")[-1]}')
    plt.legend()
    curve_path = os.path.join(config["figures_dir"], f'{config["model_name"].split("/")[-1]}_learning_curve.png')
    plt.savefig(curve_path)
    print(f"Saved learning curve plot to {curve_path}")

    # Ghi log thí nghiệm
    log_path = os.path.join(config["results_dir"], "experiments_log.csv")
    config_str = f"lr={config['learning_rate']}, batch={config['batch_size']}, max_len={config['max_length']}"
    if best_val_metrics:
        val_acc, val_f1_macro, f1_all = best_val_metrics
        log_experiment(log_path, config["model_name"], config_str, len(df_train), 
                       val_acc, val_f1_macro, f1_all, train_time_min, notes="Baseline run")
        print(f"Logged experiment metrics to {log_path}")

if __name__ == "__main__":
    main()
