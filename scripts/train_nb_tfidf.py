from pathlib import Path
import sys
import time
from datetime import date

import joblib
import pandas as pd
import yaml

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# ============================================================
# 1. Đường dẫn project
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import tokenize_for_ml_rnn


TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_PATH = PROJECT_ROOT / "data" / "processed" / "val.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
RESULT_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "nb_tfidf.pkl"
MODEL_PATH_VERSIONED = MODEL_DIR / "nb_tfidf_0708_v1.pkl"

EXPERIMENT_LOG_PATH = RESULT_DIR / "experiments_log.csv"
VAL_PRED_PATH = RESULT_DIR / "nb_tfidf_val_predictions.csv"
VAL_REPORT_PATH = RESULT_DIR / "nb_tfidf_val_report.csv"

CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# ============================================================
# 2. Cấu hình task
# ============================================================

MODEL_NAME = "NB_TFIDF"
MODEL_CONFIG = "TfidfVectorizer(max_features=20000, ngram_range=(1,2)) + MultinomialNB(alpha=1.0)"

TEXT_COL = "text_clean"
LABEL_COL = "label"

LABEL_ORDER = ["Tiêu cực", "Bình thường", "Tích cực"]


# ============================================================
# 3. Đọc train.csv và val.csv
# ============================================================

if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy train.csv: {TRAIN_PATH}")

if not VAL_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy val.csv: {VAL_PATH}")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

required_cols = {TEXT_COL, LABEL_COL}

if not required_cols.issubset(train_df.columns):
    raise ValueError(f"train.csv thiếu cột: {required_cols - set(train_df.columns)}")

if not required_cols.issubset(val_df.columns):
    raise ValueError(f"val.csv thiếu cột: {required_cols - set(val_df.columns)}")

print("Train shape:", train_df.shape)
print("Val shape:", val_df.shape)

print("\nTrain label distribution:")
print(train_df[LABEL_COL].value_counts())

print("\nVal label distribution:")
print(val_df[LABEL_COL].value_counts())


# ============================================================
# 4. Tách từ bằng underthesea cho nhánh ML
# ============================================================

print("\nĐang tách từ tiếng Việt bằng underthesea...")

start_time = time.time()

X_train = train_df[TEXT_COL].fillna("").astype(str).apply(tokenize_for_ml_rnn)
y_train = train_df[LABEL_COL].astype(str)

X_val = val_df[TEXT_COL].fillna("").astype(str).apply(tokenize_for_ml_rnn)
y_val = val_df[LABEL_COL].astype(str)

print("Tách từ xong.")


# ============================================================
# 5. Train Naïve Bayes + TF-IDF
# ============================================================

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            lowercase=False
        )
    ),
    (
        "nb",
        MultinomialNB(alpha=1.0)
    )
])

print("\nĐang train Naïve Bayes + TF-IDF...")

pipeline.fit(X_train, y_train)

train_time_min = round((time.time() - start_time) / 60, 4)

print("Train xong.")


# ============================================================
# 6. Đánh giá trên validation set
# ============================================================

y_val_pred = pipeline.predict(X_val)

val_acc = accuracy_score(y_val, y_val_pred)
val_f1_macro = f1_score(y_val, y_val_pred, average="macro", zero_division=0)

f1_per_label = f1_score(
    y_val,
    y_val_pred,
    labels=LABEL_ORDER,
    average=None,
    zero_division=0
)

val_f1_negative = f1_per_label[0]
val_f1_neutral = f1_per_label[1]
val_f1_positive = f1_per_label[2]

print("\n===== VALIDATION RESULT =====")
print(f"Accuracy: {val_acc:.4f}")
print(f"F1-macro: {val_f1_macro:.4f}")
print(f"F1-Tiêu cực: {val_f1_negative:.4f}")
print(f"F1-Bình thường: {val_f1_neutral:.4f}")
print(f"F1-Tích cực: {val_f1_positive:.4f}")

print("\nClassification report:")
print(classification_report(y_val, y_val_pred, labels=LABEL_ORDER, zero_division=0))


# ============================================================
# 7. Lưu model
# ============================================================

joblib.dump(pipeline, MODEL_PATH)
joblib.dump(pipeline, MODEL_PATH_VERSIONED)

print("\nĐã lưu model:")
print(f"- {MODEL_PATH}")
print(f"- {MODEL_PATH_VERSIONED}")


# ============================================================
# 8. Lưu kết quả dự đoán validation
# ============================================================

val_pred_df = val_df.copy()
val_pred_df["y_true"] = y_val.values
val_pred_df["y_pred"] = y_val_pred
val_pred_df["is_correct"] = val_pred_df["y_true"] == val_pred_df["y_pred"]

val_pred_df.to_csv(VAL_PRED_PATH, index=False, encoding="utf-8-sig")

report_dict = classification_report(
    y_val,
    y_val_pred,
    labels=LABEL_ORDER,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(VAL_REPORT_PATH, encoding="utf-8-sig")

print("\nĐã lưu validation outputs:")
print(f"- {VAL_PRED_PATH}")
print(f"- {VAL_REPORT_PATH}")


# ============================================================
# 9. Ghi experiments_log.csv 
# ============================================================

log_columns = [
    "date",
    "model",
    "config",
    "train_size",
    "val_acc",
    "val_f1_macro",
    "val_f1_negative",
    "val_f1_neutral",
    "val_f1_positive",
    "train_time_min",
    "notes"
]

new_log = {
    "date": str(date.today()),
    "model": MODEL_NAME,
    "config": MODEL_CONFIG,
    "train_size": int(len(train_df)),
    "val_acc": round(float(val_acc), 4),
    "val_f1_macro": round(float(val_f1_macro), 4),
    "val_f1_negative": round(float(val_f1_negative), 4),
    "val_f1_neutral": round(float(val_f1_neutral), 4),
    "val_f1_positive": round(float(val_f1_positive), 4),
    "train_time_min": train_time_min,
    "notes": "B1 baseline; train on train.csv; validate on val.csv; test.csv locked and not used"
}

if EXPERIMENT_LOG_PATH.exists():
    log_df = pd.read_csv(EXPERIMENT_LOG_PATH)
else:
    log_df = pd.DataFrame(columns=log_columns)

for col in log_columns:
    if col not in log_df.columns:
        log_df[col] = None

log_df = log_df[log_columns]
log_df = pd.concat([log_df, pd.DataFrame([new_log])], ignore_index=True)

log_df.to_csv(EXPERIMENT_LOG_PATH, index=False, encoding="utf-8-sig")

print("\nĐã ghi log:")
print(f"- {EXPERIMENT_LOG_PATH}")
print(pd.DataFrame([new_log]))


# ============================================================
# 10. Cập nhật config.yaml
# ============================================================

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
else:
    config = {}

config.setdefault("baseline", {})
config["baseline"]["nb_tfidf"] = {
    "model_path": "models/baseline/nb_tfidf.pkl",
    "versioned_model_path": "models/baseline/nb_tfidf_0708_v1.pkl",
    "vectorizer": {
        "type": "TfidfVectorizer",
        "max_features": 20000,
        "ngram_range": [1, 2]
    },
    "classifier": {
        "type": "MultinomialNB",
        "alpha": 1.0
    },
    "validation": {
        "accuracy": round(float(val_acc), 4),
        "f1_macro": round(float(val_f1_macro), 4),
        "f1_negative": round(float(val_f1_negative), 4),
        "f1_neutral": round(float(val_f1_neutral), 4),
        "f1_positive": round(float(val_f1_positive), 4)
    },
    "test_used": False
}

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
