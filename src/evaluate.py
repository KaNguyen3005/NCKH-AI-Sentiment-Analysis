import os
import argparse
import yaml
import time
import torch
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

from dataset import create_data_loader

def evaluate_predictions(model, data_loader, device):
    """
    Chạy dự đoán trên tập dữ liệu và thu thập kết quả, đo thời gian dự đoán (inference time).
    """
    model.eval()
    predictions = []
    real_values = []

    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            real_values.extend(labels.cpu().numpy())
            
    total_time_ms = (time.time() - start_time) * 1000
    avg_inference_time = total_time_ms / len(predictions) if len(predictions) > 0 else 0

    return predictions, real_values, avg_inference_time

def main():
    parser = argparse.ArgumentParser(description="Evaluate Sentiment Analysis Model")
    parser.add_argument("--config", type=str, default="../config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Cấu hình fallback
    config = {
        "model_name": "vinai/phobert-base",
        "num_classes": 3,
        "max_length": 128,
        "batch_size": 16,
        "test_data": "../data/processed/test.csv",
        "save_dir": "../models/phobert/best_checkpoint",
        "report_dir": "../report",
        "figures_dir": "../data/figures"
    }

    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config.update(yaml.safe_load(f) or {})

    os.makedirs(config["report_dir"], exist_ok=True)
    os.makedirs(config["figures_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    
    print("Loading test data...")
    try:
        df_test = pd.read_csv(config["test_data"], encoding="utf-8")
    except FileNotFoundError as e:
        print(f"Lỗi: {e}. Vui lòng đảm bảo file dữ liệu test tồn tại.")
        return

    test_loader = create_data_loader(df_test, tokenizer, config["max_length"], config["batch_size"], shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], 
        num_labels=config["num_classes"]
    )
    
    model_path = os.path.join(config["save_dir"], "best_model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded best model weights from {model_path}")
    
    model = model.to(device)
    
    # 1. Tính tổng tham số của mô hình (Params)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Tổng số tham số của mô hình (Params): {total_params:,}")

    # Chạy suy luận (Inference)
    y_pred, y_true, avg_inference_time = evaluate_predictions(model, test_loader, device)

    # 2. Tính toán các chỉ số theo yêu cầu
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    report = classification_report(y_true, y_pred, output_dict=True)
    # Giả định nhãn 1 là "Bình thường" (Tùy map nhãn của dữ liệu gốc, cần chỉnh nếu khác 1)
    neu_score = report.get('1', {}).get('f1-score', 0) if '1' in report else 0
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1-Macro: {f1_macro:.4f}")
    print(f"NEU score (Nhãn Bình thường): {neu_score:.4f}")
    print(f"Inference Time: {avg_inference_time:.2f} ms/câu")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # 3. Lưu metrics vào file JSON
    metrics_path = os.path.join(config["report_dir"], "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy, 
            "f1_macro": f1_macro,
            "neu_score": neu_score,
            "inference_time_ms_per_sentence": avg_inference_time,
            "total_parameters": total_params,
            "classification_report": report
        }, f, indent=4, ensure_ascii=False)
    print(f"Metrics saved to {metrics_path}")

    # 4. Lưu lại vector output để phục vụ Error Analysis và McNemar Test sau này
    df_test['predicted_label'] = y_pred
    df_test['true_label'] = y_true
    mcnemar_path = os.path.join(config["report_dir"], "test_predictions_for_error_analysis.csv")
    df_test.to_csv(mcnemar_path, index=False, encoding="utf-8")
    print(f"Saved predictions for McNemar/Error Analysis at {mcnemar_path}")

    # 5. Vẽ và lưu biểu đồ Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Dự đoán (Predicted Labels)')
    plt.ylabel('Thực tế (True Labels)')
    plt.title('Confusion Matrix')
    
    cm_path = os.path.join(config["figures_dir"], "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix plot saved to {cm_path}")

if __name__ == "__main__":
    main()
