from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np
import yaml

# ============================================================
# 1. Khai báo đường dẫn
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SRC_DIR = PROJECT_ROOT / "src"
RESULT_DIR = PROJECT_ROOT / "results"

MAIN_PATH = RAW_DIR / "khao_sat_AI_NLP_1000.csv"
UIT_PATH = RAW_DIR / "uit-vsfc" / "uit_vsfc_all.csv"

MERGED_PATH = PROCESSED_DIR / "merged.csv"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
REPORT_PATH = RESULT_DIR / "merge_report.json"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Cho phép import src/preprocessing.py

sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import clean_text, prepare_for_plm
# ============================================================
# 2. Mapping nhãn về đúng 3 lớp
# ============================================================

LABEL_TO_ID = {
    "Tiêu cực": 0,
    "Bình thường": 1,
    "Tích cực": 2,
}

ID_TO_LABEL = {
    0: "Tiêu cực",
    1: "Bình thường",
    2: "Tích cực",
}

LABEL_NORMALIZE = {
    # Negative
    "0": "Tiêu cực",
    "negative": "Tiêu cực",
    "neg": "Tiêu cực",
    "tiêu cực": "Tiêu cực",
    "tieu cuc": "Tiêu cực",

    # Neutral
    "1": "Bình thường",
    "neutral": "Bình thường",
    "neu": "Bình thường",
    "bình thường": "Bình thường",
    "binh thuong": "Bình thường",
    "trung lập": "Bình thường",
    "trung lap": "Bình thường",

    # Positive
    "2": "Tích cực",
    "positive": "Tích cực",
    "pos": "Tích cực",
    "tích cực": "Tích cực",
    "tich cuc": "Tích cực",
}


def normalize_label(value):
    """Chuẩn hóa mọi dạng nhãn về 3 nhãn tiếng Việt."""
    if pd.isna(value):
        return None

    key = str(value).strip().lower()
    return LABEL_NORMALIZE.get(key, str(value).strip())


# ============================================================
# 3. Kiểm tra file đầu vào
# ============================================================

if not MAIN_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy file khảo sát chính: {MAIN_PATH}")

if not UIT_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy file UIT-VSFC đã gộp: {UIT_PATH}")


# ============================================================
# 4. Đọc bộ khảo sát chính
# ============================================================

df_main = pd.read_csv(MAIN_PATH)

required_main_cols = {"nhan_xet", "nhan", "nhan_so"}
missing_main_cols = required_main_cols - set(df_main.columns)

if missing_main_cols:
    raise ValueError(f"File khảo sát chính thiếu cột: {missing_main_cols}")

df_main_std = df_main.rename(columns={
    "nhan_xet": "text",
    "nhan": "label",
    "nhan_so": "label_id",
}).copy()

df_main_std["source"] = "Khao_sat_AI"
df_main_std["original_split"] = "main"

df_main_std = df_main_std[[
    "text",
    "label",
    "label_id",
    "source",
    "original_split"
]]


# ============================================================
# 5. Đọc UIT-VSFC
# ============================================================

df_uit = pd.read_csv(UIT_PATH)

required_uit_cols = {"text", "label", "label_id", "source", "original_split"}
missing_uit_cols = required_uit_cols - set(df_uit.columns)

if missing_uit_cols:
    raise ValueError(f"File UIT-VSFC thiếu cột: {missing_uit_cols}")

df_uit_std = df_uit[[
    "text",
    "label",
    "label_id",
    "source",
    "original_split"
]].copy()


# ============================================================
# 6. Gộp 2 nguồn dữ liệu
# ============================================================

df_all = pd.concat([df_main_std, df_uit_std], ignore_index=True)

before_clean = len(df_all)

# Chuẩn hóa text
df_all["text"] = df_all["text"].astype(str).str.strip()

# Chuẩn hóa label về 3 nhãn
df_all["label"] = df_all["label"].apply(normalize_label)

# Xóa dòng thiếu text/label
df_all = df_all.dropna(subset=["text", "label"])

# Xóa text rỗng hoặc nan dạng chuỗi
df_all = df_all[
    (df_all["text"].str.len() > 0) &
    (df_all["text"].str.lower() != "nan")
]

# Chỉ giữ 3 nhãn hợp lệ
df_all = df_all[df_all["label"].isin(LABEL_TO_ID.keys())].copy()

after_label_filter = len(df_all)

# Tạo label_id chuẩn
df_all["label_id"] = df_all["label"].map(LABEL_TO_ID).astype(int)


# ============================================================
# 7. Áp dụng preprocessing.py
# ============================================================

df_all["text_clean"] = df_all["text"].apply(clean_text)

# Nhánh PLM: PhoBERT/mBERT dùng tokenizer riêng, không underthesea ở đây
df_all["text_plm"] = df_all["text"].apply(prepare_for_plm)

# Xóa text sau clean nếu bị rỗng
df_all = df_all[df_all["text_clean"].str.len() > 0].copy()

# Đếm số từ sau clean
df_all["word_count"] = df_all["text_clean"].str.split().str.len()


# ============================================================
# 8. Xử lý trùng lặp
# ============================================================

before_dedup = len(df_all)

# Xóa trùng chính xác theo text_clean + label
df_all = df_all.drop_duplicates(subset=["text_clean", "label"]).reset_index(drop=True)

after_dedup = len(df_all)
removed_duplicates = before_dedup - after_dedup


# ============================================================
# 9. Chốt max_length theo percentile 95
# ============================================================

p95_word_count = float(df_all["word_count"].quantile(0.95))
max_length_words_p95 = int(np.ceil(p95_word_count))

# p95 đang tính theo số từ.
# Với PhoBERT/mBERT, tokenizer tính theo subword token, nên chọn 64 để an toàn.
if max_length_words_p95 <= 64:
    model_max_length = 64
elif max_length_words_p95 <= 128:
    model_max_length = 128
else:
    model_max_length = 256


# ============================================================
# 10. Sắp xếp lại cột và lưu merged.csv
# ============================================================

df_all = df_all[[
    "text",
    "text_clean",
    "text_plm",
    "label",
    "label_id",
    "word_count",
    "source",
    "original_split"
]]

df_all.to_csv(MERGED_PATH, index=False, encoding="utf-8-sig")


# ============================================================
# 11. Tạo config.yaml
# ============================================================

label_count = df_all["label"].value_counts().to_dict()
label_percent = (
    df_all["label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .to_dict()
)

config = {
    "project": {
        "name": "Vietnamese AI Sentiment Analysis",
        "task": "3-class sentiment classification",
        "language": "vi",
    },
    "data": {
        "raw_main_path": "data/raw/khao_sat_AI_NLP_1000.csv",
        "raw_uit_vsfc_path": "data/raw/uit-vsfc/uit_vsfc_all.csv",
        "processed_path": "data/processed/merged.csv",

        "text_column": "text_clean",
        "plm_text_column": "text_plm",
        "label_column": "label",
        "label_id_column": "label_id",

        "num_labels": 3,
        "id_to_label": ID_TO_LABEL,
        "label_to_id": LABEL_TO_ID,

        "total_samples": int(len(df_all)),
        "main_samples": int((df_all["source"] == "Khao_sat_AI").sum()),
        "uit_vsfc_samples": int((df_all["source"] == "UIT-VSFC").sum()),

        "label_count": label_count,
        "label_distribution_percent": label_percent,

        "removed_invalid_or_empty": int(before_clean - after_label_filter),
        "removed_duplicates": int(removed_duplicates),

        "avg_word_count": float(round(df_all["word_count"].mean(), 2)),
        "p95_word_count": float(round(p95_word_count, 2)),
        "max_length_words_p95": int(max_length_words_p95),
    },
    "model": {
        "phobert_name": "vinai/phobert-base",
        "mbert_name": "bert-base-multilingual-cased",
        "max_length": int(model_max_length),
    },
    "training": {
        "split_ratio": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
        },
        "random_seeds": [42, 2024, 2026],
        "use_class_weight": True,
        "use_focal_loss": True,
    },
    "notes": {
        "max_length_explanation": (
            f"Percentile 95 theo số từ là {max_length_words_p95}. "
            f"Vì PhoBERT/mBERT dùng subword tokenizer, chọn model.max_length = {model_max_length}."
        ),
        "task_06_07": (
            "Gộp khảo sát chính và UIT-VSFC về cùng format 3 nhãn, "
            "tạo merged.csv và chốt max_length vào config.yaml."
        ),
    },
}

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


# ============================================================
# 12. Lưu report JSON phụ để kiểm tra
# ============================================================

report = {
    "main_input_shape": list(df_main.shape),
    "uit_input_shape": list(df_uit.shape),
    "before_clean": int(before_clean),
    "after_label_filter": int(after_label_filter),
    "before_dedup": int(before_dedup),
    "after_dedup": int(after_dedup),
    "removed_duplicates": int(removed_duplicates),
    "merged_path": str(MERGED_PATH),
    "config_path": str(CONFIG_PATH),
    "label_count": label_count,
    "label_percent": label_percent,
    "p95_word_count": p95_word_count,
    "max_length_words_p95": max_length_words_p95,
    "model_max_length": model_max_length,
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)


# ============================================================
# 13. In kết quả kiểm tra
# ============================================================

print("===== TASK 06/07 DONE: MERGE DATA =====")
print(f"Main input shape: {df_main.shape}")
print(f"UIT-VSFC input shape: {df_uit.shape}")
print(f"Before clean: {before_clean}")
print(f"After label filter: {after_label_filter}")
print(f"Before dedup: {before_dedup}")
print(f"After dedup: {after_dedup}")
print(f"Removed duplicates: {removed_duplicates}")

print("\nSaved files:")
print(f"- {MERGED_PATH}")
print(f"- {CONFIG_PATH}")
print(f"- {REPORT_PATH}")

print("\nLabel count:")
print(df_all["label"].value_counts())

print("\nLabel percent:")
print(df_all["label"].value_counts(normalize=True).mul(100).round(2))

print("\nWord count describe:")
print(df_all["word_count"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

print(f"\nmax_length_words_p95 = {max_length_words_p95}")
print(f"model.max_length = {model_max_length}")