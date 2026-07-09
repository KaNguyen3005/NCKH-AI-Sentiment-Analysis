from pathlib import Path
import json
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


# ============================================================
# 1. Khai báo đường dẫn
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MERGED_PATH = PROCESSED_DIR / "merged.csv"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

TRAIN_PATH = PROCESSED_DIR / "train.csv"
VAL_PATH = PROCESSED_DIR / "val.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
SEEDS_PATH = PROCESSED_DIR / "seeds.json"
SPLIT_REPORT_PATH = PROJECT_ROOT / "results" / "split_report.json"

PROJECT_ROOT.joinpath("results").mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Cấu hình split
# ============================================================

SEEDS = [42, 2024, 2026]
PRIMARY_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

LABEL_COL = "label"
LABEL_ID_COL = "label_id"


# ============================================================
# 3. Đọc merged.csv
# ============================================================

if not MERGED_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy file: {MERGED_PATH}")

df = pd.read_csv(MERGED_PATH)

required_cols = {
    "text",
    "text_clean",
    "text_plm",
    "label",
    "label_id",
    "word_count",
    "source",
    "original_split"
}

missing_cols = required_cols - set(df.columns)

if missing_cols:
    raise ValueError(f"merged.csv thiếu các cột bắt buộc: {missing_cols}")

# Tạo sample_id cố định để truy vết sau này
df = df.reset_index(drop=True)
df.insert(0, "sample_id", range(len(df)))

print("Merged shape:", df.shape)
print("\nLabel distribution full data:")
print(df[LABEL_COL].value_counts())
print(df[LABEL_COL].value_counts(normalize=True).mul(100).round(2))


# ============================================================
# 4. Hàm split 70/15/15 có stratify
# ============================================================

def make_stratified_split(dataframe: pd.DataFrame, seed: int):
    """
    Split 70/15/15 theo stratified label_id.

    Bước 1:
    - Tách train = 70%
    - temp = 30%

    Bước 2:
    - Chia temp thành val = 15%, test = 15%
    - Vì temp đang là 30%, nên val/test = 50/50 trong temp
    """

    train_df, temp_df = train_test_split(
        dataframe,
        test_size=(1 - TRAIN_RATIO),
        random_state=seed,
        stratify=dataframe[LABEL_ID_COL],
        shuffle=True
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df[LABEL_ID_COL],
        shuffle=True
    )

    train_df = train_df.sort_values("sample_id").reset_index(drop=True)
    val_df = val_df.sort_values("sample_id").reset_index(drop=True)
    test_df = test_df.sort_values("sample_id").reset_index(drop=True)

    return train_df, val_df, test_df


# ============================================================
# 5. Tạo split chính thức bằng seed 42
# ============================================================

train_df, val_df, test_df = make_stratified_split(df, PRIMARY_SEED)

train_df.to_csv(TRAIN_PATH, index=False, encoding="utf-8-sig")
val_df.to_csv(VAL_PATH, index=False, encoding="utf-8-sig")
test_df.to_csv(TEST_PATH, index=False, encoding="utf-8-sig")


# ============================================================
# 6. Kiểm tra không trùng sample_id giữa train/val/test
# ============================================================

train_ids = set(train_df["sample_id"])
val_ids = set(val_df["sample_id"])
test_ids = set(test_df["sample_id"])

assert train_ids.isdisjoint(val_ids), "Lỗi: train và val bị trùng sample_id"
assert train_ids.isdisjoint(test_ids), "Lỗi: train và test bị trùng sample_id"
assert val_ids.isdisjoint(test_ids), "Lỗi: val và test bị trùng sample_id"

assert len(train_df) + len(val_df) + len(test_df) == len(df), (
    "Lỗi: tổng train + val + test không bằng merged"
)


# ============================================================
# 7. Tạo seeds.json cho 3 seed
# ============================================================

seeds_info = {
    "description": (
        "Stratified split 70/15/15. "
        "train.csv, val.csv, test.csv are generated using primary_seed=42. "
        "The indices for all 3 seeds are stored for reproducibility."
    ),
    "primary_seed": PRIMARY_SEED,
    "seeds": SEEDS,
    "split_ratio": {
        "train": TRAIN_RATIO,
        "validation": VAL_RATIO,
        "test": TEST_RATIO
    },
    "stratify_by": LABEL_ID_COL,
    "locked_test_set": True,
    "locked_note": (
        "test.csv must not be used for tuning. "
        "Use train.csv for training and val.csv for model selection. "
        "Run test.csv only once in final evaluation stage."
    ),
    "splits": {}
}

for seed in SEEDS:
    tr, va, te = make_stratified_split(df, seed)

    seeds_info["splits"][str(seed)] = {
        "train_size": int(len(tr)),
        "val_size": int(len(va)),
        "test_size": int(len(te)),
        "train_sample_ids": tr["sample_id"].astype(int).tolist(),
        "val_sample_ids": va["sample_id"].astype(int).tolist(),
        "test_sample_ids": te["sample_id"].astype(int).tolist(),
        "label_distribution": {
            "train": tr[LABEL_COL].value_counts().to_dict(),
            "val": va[LABEL_COL].value_counts().to_dict(),
            "test": te[LABEL_COL].value_counts().to_dict(),
        }
    }

with open(SEEDS_PATH, "w", encoding="utf-8") as f:
    json.dump(seeds_info, f, ensure_ascii=False, indent=2)


# ============================================================
# 8. Tạo split report để kiểm tra nhanh
# ============================================================

def split_summary(split_name: str, split_df: pd.DataFrame):
    count = split_df[LABEL_COL].value_counts().to_dict()
    percent = split_df[LABEL_COL].value_counts(normalize=True).mul(100).round(2).to_dict()

    return {
        "name": split_name,
        "size": int(len(split_df)),
        "percent_of_total": round(len(split_df) / len(df) * 100, 2),
        "label_count": count,
        "label_percent": percent,
        "source_count": split_df["source"].value_counts().to_dict()
    }


split_report = {
    "total_samples": int(len(df)),
    "primary_seed": PRIMARY_SEED,
    "train": split_summary("train", train_df),
    "validation": split_summary("validation", val_df),
    "test": split_summary("test", test_df),
    "output_files": {
        "train": str(TRAIN_PATH),
        "validation": str(VAL_PATH),
        "test": str(TEST_PATH),
        "seeds": str(SEEDS_PATH)
    }
}

with open(SPLIT_REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(split_report, f, ensure_ascii=False, indent=2)


# ============================================================
# 9. Cập nhật config.yaml
# ============================================================

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
else:
    config = {}

config.setdefault("data", {})
config["data"]["train_path"] = "data/processed/train.csv"
config["data"]["val_path"] = "data/processed/val.csv"
config["data"]["test_path"] = "data/processed/test.csv"
config["data"]["seeds_path"] = "data/processed/seeds.json"

config.setdefault("training", {})
config["training"]["primary_seed"] = PRIMARY_SEED
config["training"]["random_seeds"] = SEEDS
config["training"]["split_ratio"] = {
    "train": TRAIN_RATIO,
    "validation": VAL_RATIO,
    "test": TEST_RATIO
}
config["training"]["stratify_by"] = LABEL_ID_COL
config["training"]["test_set_locked"] = True

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


# ============================================================
# 10. In kết quả
# ============================================================

print("\n===== TASK 07/07 DONE: STRATIFIED SPLIT =====")
print(f"Total samples: {len(df)}")
print(f"Train: {len(train_df)} | {len(train_df) / len(df) * 100:.2f}%")
print(f"Val:   {len(val_df)} | {len(val_df) / len(df) * 100:.2f}%")
print(f"Test:  {len(test_df)} | {len(test_df) / len(df) * 100:.2f}%")

print("\nTrain label distribution:")
print(train_df[LABEL_COL].value_counts())
print(train_df[LABEL_COL].value_counts(normalize=True).mul(100).round(2))

print("\nVal label distribution:")
print(val_df[LABEL_COL].value_counts())
print(val_df[LABEL_COL].value_counts(normalize=True).mul(100).round(2))

print("\nTest label distribution:")
print(test_df[LABEL_COL].value_counts())
print(test_df[LABEL_COL].value_counts(normalize=True).mul(100).round(2))

print("\nSaved files:")
print(f"- {TRAIN_PATH}")
print(f"- {VAL_PATH}")
print(f"- {TEST_PATH}")
print(f"- {SEEDS_PATH}")
print(f"- {SPLIT_REPORT_PATH}")
print(f"- {CONFIG_PATH}")

print("\nLOCK NOTE:")
print("test.csv đã được tạo và phải khóa từ đây. Không dùng test.csv để tune model.")