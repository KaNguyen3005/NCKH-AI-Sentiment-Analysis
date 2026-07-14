# NCKH-AI-Sentiment-Analysis

Dự án Nghiên cứu Khoa học: Phân tích cảm xúc (Sentiment Analysis) trên văn bản tiếng Việt. 

Dự án này tập trung vào việc nghiên cứu, triển khai và đánh giá hiệu năng của nhiều phương pháp khác nhau trong bài toán phân tích cảm xúc tiếng Việt. Các phương pháp trải dài từ các mô hình học máy truyền thống, học sâu (Deep Learning) cho đến việc ứng dụng các Mô hình Ngôn ngữ Lớn (LLMs).

## 🚀 Giới thiệu (Introduction)

Mục tiêu chính của nghiên cứu là đánh giá tính hiệu quả của các kiến trúc mô hình khác nhau khi xử lý văn bản tiếng Việt. Các tập dữ liệu chính được sử dụng bao gồm:
- **UIT-VSFC** (Vietnamese Students' Feedback Corpus)
- **Dữ liệu tự thu thập** (`khao_sat_AI_NLP_1000.csv`)

**Các phương pháp được thực nghiệm và so sánh:**
1. **Machine Learning:** Naïve Bayes, SVM kết hợp với TF-IDF.
2. **Deep Learning:** BiLSTM, GRU sử dụng embedding tùy chỉnh (FastText).
3. **Transformer-based:** PhoBERT, mBERT (Multilingual BERT).
4. **Large Language Models (LLMs):** Đánh giá Zero-shot và Few-shot sử dụng OpenAI (GPT) và Google Generative AI (Gemini).

## 📁 Cấu trúc thư mục (Project Structure)

```text
NCKH-AI-Sentiment-Analysis/
├── app/               # Web App Demo xây dựng bằng Streamlit
├── data/              # Chứa dữ liệu (raw data: uit-vsfc, dữ liệu khảo sát) và biểu đồ (figures)
├── docs/              # Tài liệu tham khảo, hướng dẫn của dự án
├── notebooks/         # Các Jupyter Notebook dùng để thử nghiệm và phân tích
│   ├── 01_EDA.ipynb
│   ├── 02_baseline_NB_SVM.ipynb
│   ├── 03_baseline_BiLSTM_GRU.ipynb
│   ├── 04_baseline_mBERT.ipynb
│   ├── 05_phobert_training.ipynb
│   ├── 06_evaluation.ipynb
│   └── 07_llm_baseline.ipynb
├── report/            # Báo cáo kết quả nghiên cứu
├── results/           # Kết quả thí nghiệm, file CSV (cv_results, llm_predictions...)
├── scripts/           # Các bash script hoặc script chạy tự động
├── src/               # Mã nguồn chính của dự án
│   ├── models/        # Định nghĩa các kiến trúc mô hình (BiLSTM, GRU...)
│   ├── dataset.py     # Xử lý dữ liệu và DataLoader
│   ├── evaluate.py    # Các hàm đánh giá và vẽ biểu đồ (Confusion matrix, Learning curve)
│   ├── predict.py     # Script dự đoán cảm xúc cho dữ liệu mới
│   ├── preprocessing.py # Tiền xử lý văn bản tiếng Việt (tách từ với underthesea)
│   └── train.py       # Vòng lặp huấn luyện (Training loop)
├── config.yaml        # Cấu hình siêu tham số và đường dẫn
├── requirements.txt   # Danh sách các thư viện cần cài đặt
└── README.md          # Tài liệu tổng quan dự án (file này)
```

## 🛠 Công nghệ & Thư viện (Tech Stack)

- **Ngôn ngữ:** Python
- **Xử lý ngôn ngữ tự nhiên (NLP):** `underthesea`, `fasttext-wheel`, `transformers`
- **Học máy (Machine Learning):** `scikit-learn`, `statsmodels`
- **Học sâu (Deep Learning):** `torch` (PyTorch)
- **Tối ưu siêu tham số (Hyperparameter Tuning):** `optuna`
- **API LLMs:** `openai`, `google-generativeai`
- **Ứng dụng Web (Web App Demo):** `streamlit`
- **Trực quan hóa (Visualization):** `matplotlib`, `seaborn`, `wordcloud`

## ⚙️ Cài đặt & Sử dụng (Installation & Usage)

**Bước 1: Clone kho lưu trữ**
```bash
git clone https://github.com/KaNguyen3005/NCKH-AI-Sentiment-Analysis.git
cd NCKH-AI-Sentiment-Analysis
```

**Bước 2: Cài đặt các thư viện phụ thuộc**
Khuyến nghị sử dụng môi trường ảo (virtual environment) để tránh xung đột thư viện:
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường (trên Windows)
venv\Scripts\activate     
# Kích hoạt môi trường (trên Linux/Mac)
source venv/bin/activate  

# Cài đặt thư viện
pip install -r requirements.txt
```

**Bước 3: Chạy ứng dụng Demo (Streamlit Web App)**
```bash
streamlit run app/app.py
```

## 📊 Thử nghiệm & Kết quả (Experiments & Results)

Chi tiết về quá trình huấn luyện và đánh giá mô hình được lưu trữ ở thư mục `notebooks/`. Bạn có thể chạy tuần tự từ file `01_EDA.ipynb` (Phân tích dữ liệu thăm dò) cho đến file `07_llm_baseline.ipynb` (Đánh giá mô hình LLM).

Kết quả từ các mô hình, bao gồm độ chính xác (Accuracy), F1-score và các kiểm định thống kê (McNemar test) được tự động lưu trữ và tổng hợp tại thư mục `results/`.

## 📑 Báo cáo Thực nghiệm (Experiment Reports)

- [Báo cáo Kỹ thuật Xử lý Imbalance Data (Focal Loss vs Class Weight)](docs/01_imbalance_report.md)
- [Báo cáo So sánh Kiến trúc PhoBERT Head (CLS vs Mean-Pooling)](docs/02_head_architecture_report.md)
