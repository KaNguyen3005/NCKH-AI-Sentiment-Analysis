from pathlib import Path
import pandas as pd
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "uit-vsfc"
RAW_DIR.mkdir(parents=True, exist_ok=True)

print("Đang tải UIT-VSFC từ Hugging Face...")

dataset = load_dataset(
    "uitnlp/vietnamese_students_feedback",
    trust_remote_code=True
)

sentiment_map = {
    0: "Tiêu cực",
    1: "Bình thường",
    2: "Tích cực"
}

all_dfs = []

for split_name, split_data in dataset.items():
    df = split_data.to_pandas()

    df = df.rename(columns={
        "sentence": "text",
        "sentiment": "label",
        "topic": "topic"
    })

    df["label_id"] = df["label"]
    df["label"] = df["label_id"].map(sentiment_map)

    df["source"] = "UIT-VSFC"
    df["original_split"] = split_name

    df = df[["text", "label", "label_id", "topic", "source", "original_split"]]

    output_path = RAW_DIR / f"{split_name}.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Đã lưu {split_name}: {output_path} | {len(df)} dòng")

    all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True)

all_path = RAW_DIR / "uit_vsfc_all.csv"
df_all.to_csv(all_path, index=False, encoding="utf-8-sig")

print("\nTải xong UIT-VSFC.")
print(f"Tổng số dòng: {len(df_all)}")
print(f"File gộp: {all_path}")

print("\nPhân phối nhãn:")
print(df_all["label"].value_counts())