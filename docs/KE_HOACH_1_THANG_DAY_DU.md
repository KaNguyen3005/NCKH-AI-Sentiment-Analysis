# ⏱️ KẾ HOẠCH 1 THÁNG — BẢN ĐẦY ĐỦ NHẤT
## Phân tích cảm xúc văn bản tiếng Việt về AI (PhoBERT) · 03/07 → 01/08/2026

**Nhóm:** Mô Ha Mách Bu Ba Ka (N23DCCN164 — Model/Lý thuyết) · Thái Ngọc Thi (N23DCCN126 — Data/Đánh giá) · Lê Đình Bảo (N23DCCN141 — Hạ tầng/Viết) | Lớp D23CQCNO3-N

> File này = `KE_HOACH_1_THANG.md` (lịch theo ngày) + toàn bộ phần nền tảng của roadmap gốc (cấu trúc thư mục, công nghệ, danh sách paper, rủi ro, bản đồ section) đã bị lược khi nén thời gian. Dùng file này làm bản duy nhất, không cần mở lại 2 file kia.

---

## MỤC LỤC
1. Bài toán & mục tiêu
2. Cấu trúc thư mục repo
3. Công nghệ sử dụng
4. Danh sách paper bắt buộc đọc
5. Chiến lược nén thời gian & ký hiệu ưu tiên
6. **Lịch chi tiết theo ngày (Tuần 1–4)**
7. Bản đồ nội dung → `sn-article.tex`
8. Bảng kết quả mẫu (điền dần khi có số liệu)
9. Rủi ro & phương án dự phòng
10. Phân công tổng hợp theo luồng
11. Checklist "hoàn thành tối thiểu"

---

## 1. BÀI TOÁN & MỤC TIÊU

**Bài toán:** câu nhận xét tiếng Việt về công cụ AI → phân vào 1 trong 3 nhãn: Tích cực / Bình thường / Tiêu cực.

**Mô hình chính:** PhoBERT-base (`vinai/phobert-base`) fine-tuning.

**Baseline so sánh (6 mô hình + đối chứng LLM):** Naïve Bayes · SVM (LinearSVC) · BiLSTM · GRU · mBERT · **PhoBERT** + đối chứng **LLM (GPT/Gemini, zero-shot & few-shot)** — đáp ứng yêu cầu đề tài về "mô hình ngôn ngữ lớn".

**Đóng góp khoa học riêng (bắt buộc phải có để không chỉ là "áp dụng lại PhoBERT"):**
> So sánh **(a)** classification head: CLS token vs mean-pooling (vs attention pooling nếu kịp) và **(b)** chiến lược xử lý mất cân bằng nhãn: class-weight vs Focal Loss.

**Mục tiêu số liệu:**

| Chỉ số | Mục tiêu |
|---|---|
| Accuracy | ≥ 82% |
| F1-macro | ≥ 0.75 |
| NEU score (F1 lớp Bình thường) | báo cáo riêng — cần chốt định nghĩa với GV ngay 03/07 |
| PhoBERT vs baseline tốt nhất | ≥ +5% F1, có kiểm định McNemar |

**Quyết định dữ liệu (chốt luôn, không bàn lại):** gộp thêm UIT-VSFC vào bộ 1.000 mẫu ngay Tuần 1 (vì nhãn Tiêu cực gốc chỉ ~7%), dùng 3-seed thay 5-fold CV để tiết kiệm thời gian nhưng vẫn có mean±std.

---

## 2. CẤU TRÚC THƯ MỤC REPO

Tạo đúng cấu trúc này **ở Thứ 6 03/07** (task đầu tiên), toàn bộ output về sau đổ vào đây:

```
nlp-sentiment-research/
├── data/
│   ├── raw/                        # gốc + uit-vsfc/ — KHÔNG sửa trực tiếp
│   │   ├── khao_sat_AI_NLP_1000.csv
│   │   └── uit-vsfc/
│   ├── processed/                  # merged.csv, train/val/test.csv
│   │   ├── merged.csv
│   │   ├── train.csv · val.csv · test.csv   🔒 khóa sau 07/07
│   │   └── seeds.json              # 3 seed split thay vì folds.json
│   └── figures/                    # biểu đồ EDA, learning curve, confusion matrix
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_baseline_NB_SVM.ipynb
│   ├── 03_baseline_BiLSTM_GRU.ipynb
│   ├── 04_baseline_mBERT.ipynb
│   ├── 05_phobert_training.ipynb
│   ├── 06_evaluation.ipynb
│   └── 07_llm_baseline.ipynb
├── src/
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── models/
│   │   ├── bilstm.py · gru.py · phobert_heads.py
│   ├── train.py · evaluate.py · predict.py
├── models/
│   ├── baseline/                   # nb_tfidf.pkl, svm_tfidf.pkl, bilstm_best.pt, gru_best.pt, mbert_finetuned/
│   └── phobert/best_checkpoint/
├── results/
│   ├── experiments_log.csv         # log mọi lần train
│   ├── cv_results.csv              # kết quả 3-seed
│   └── llm_predictions.csv
├── app/
│   └── app.py                      # demo Streamlit
├── docs/
│   └── related_works.md            # tóm tắt paper
├── report/
│   ├── sn-article.tex              # paper chính
│   ├── sn-jnl.cls
│   └── Mau_baocao.tex              # slide Beamer
├── config.yaml
├── requirements.txt
└── README.md
```

**Task tạo cấu trúc:** Lê Đình Bảo, `mkdir` toàn bộ cây thư mục + `git init` + push commit đầu tiên trong 03/07.

---

## 3. CÔNG NGHỆ SỬ DỤNG

| Nhóm | Công cụ |
|---|---|
| Ngôn ngữ & framework | Python 3.10+, PyTorch 2.x |
| NLP | HuggingFace `transformers` (`vinai/phobert-base`, `bert-base-multilingual-cased`), `underthesea` (tách từ) |
| Embedding | FastText `cc.vi.300` (cho BiLSTM/GRU) |
| ML truyền thống | scikit-learn (TF-IDF, Naïve Bayes, LinearSVC) |
| Thống kê | `statsmodels` (McNemar test) |
| Tuning | Optuna (nếu kịp), thủ công grid nhỏ nếu không |
| Theo dõi huấn luyện | TensorBoard hoặc log thủ công vào CSV |
| Trực quan hóa | matplotlib, seaborn, wordcloud |
| LLM đối chứng | OpenAI API (GPT) hoặc Google AI Studio (Gemini) |
| Demo | Streamlit (chạy local là đủ, deploy HF Spaces nếu kịp) |
| Hạ tầng tính toán | Google Colab T4 hoặc Kaggle P100 (free tier) |
| Quản lý mã nguồn | GitHub, chia nhánh theo người/luồng |

---

## 4. DANH SÁCH PAPER BẮT BUỘC ĐỌC (9 bài + 2 bài thêm nếu kịp)

Đọc rải trong Tuần 1, mỗi người phụ trách 3 bài, tóm tắt theo khuôn 5 gạch đầu dòng (xem `HUONG_DAN_CHI_TIET_TASK.md` mục 1).

| # | Paper | Vai trò trong đề tài | Nguồn |
|---|---|---|---|
| 1 | Vaswani et al. (2017) — Attention Is All You Need | Nền tảng Transformer | arxiv.org/abs/1706.03762 |
| 2 | Devlin et al. (2019) — BERT | Cơ chế MLM, nền của PhoBERT | arxiv.org/abs/1810.04805 |
| 3 | Hochreiter & Schmidhuber (1997) — LSTM | Nền tảng RNN cho BiLSTM/GRU | Neural Computation |
| 4 | Kim (2014) — CNN for Sentence Classification | Tham chiếu baseline DL kinh điển | arxiv.org/abs/1408.5882 |
| 5 | **Nguyen & Nguyen (2020) — PhoBERT** | **Mô hình chính** | Findings EMNLP 2020 / arxiv 2003.00744 |
| 6 | Nguyen et al. (2018) — UIT-VSFC | Nguồn dữ liệu bổ sung | ieeexplore.ieee.org/document/8573337 |
| 7 | Nguyen P.X. et al. (2018) — DL vs traditional trên UIT-VSFC | Sát cấu trúc thí nghiệm nhóm | NICS 2018 |
| 8 | Nguyen et al. (2021) — Vietnamese Complaint Detection | Ví dụ fine-tune PhoBERT thực tế | arxiv.org/abs/2104.11969 |
| 9 | NEU-ESC — Educational Sentiment Classification | Benchmark BERT + LLM tiếng Việt | huggingface.co/datasets/hung20gg/NEU-ESC |
| 10 🟢 | Liu et al. (2019) — RoBERTa | Backbone của PhoBERT (đọc nếu kịp) | arxiv.org/abs/1907.11692 |
| 11 🟢 | Conneau et al. (2020) — XLM-R | Đối chứng đa ngữ (đọc nếu kịp) | arxiv.org/abs/1911.02116 |

---

## 5. CHIẾN LƯỢC NÉN & KÝ HIỆU ƯU TIÊN

**Cắt so với roadmap 14 tuần gốc:** 5-fold CV → 3-seed; đọc-hết-trước → đọc-song-song-lúc-làm; deploy HF Spaces và attention-pooling head chuyển xuống P3 (cắt được nếu kẹt).

| Ký hiệu | Ý nghĩa |
|---|---|
| 🔴 P1 | Sống còn — thiếu là không nộp được |
| 🟡 P2 | Quan trọng — làm để đủ chất lượng, trễ thì rút gọn |
| 🟢 P3 | Tốt nếu có — cắt không tiếc khi kẹt thời gian |

**3 luồng chạy song song bắt buộc:**
- **DATA** → Thái Ngọc Thi
- **MODEL** → Mô Ha Mách Bu Ba Ka
- **HẠ TẦNG + VIẾT** → Lê Đình Bảo

---

## 6. LỊCH CHI TIẾT THEO NGÀY

### 📅 TUẦN 1 (03/07 – 09/07): Nền tảng + Dữ liệu xong sớm
> Gate cuối tuần: repo chạy được, data sạch & split xong, 2 baseline ML xong, lý thuyết 50%.

| Ngày | Task | Ưu tiên | Người | Đầu ra |
|---|---|---|---|---|
| **T6 03/07** | Họp chốt phạm vi + tạo repo theo đúng cấu trúc mục 2 + `requirements.txt` | 🔴 P1 | Cả nhóm/Bảo | Repo, cả nhóm clone chạy |
| 03/07 | Xác nhận với GV cách hiểu "NEU score" | 🔴 P1 | Ba Ka | Ghi chú chốt định nghĩa |
| **T7 04/07** | Tải bộ chính + UIT-VSFC vào `data/raw/`; EDA nhanh (nhãn, độ dài, trùng lặp) | 🔴 P1 | Thi | `01_EDA.ipynb`, 5 biểu đồ |
| 04/07 | Đọc + tóm tắt paper #1, #2, #5 (Transformer, BERT, PhoBERT) | 🔴 P1 | Ba Ka | 3 mục `related_works.md` |
| **CN 05/07** | `src/preprocessing.py` (7 bước NFC→clean→tokenize, xem sổ tay mục 3) | 🔴 P1 | Thi + Bảo | Chạy test 3 câu mẫu |
| 05/07 | Đọc + tóm tắt paper #6, #7, #8 (UIT-VSFC, DL vs ML, Complaint) | 🟡 P2 | Bảo | 3 mục `related_works.md` |
| **T2 06/07** | Gộp UIT-VSFC về 3 nhãn → `data/processed/merged.csv`; chốt `max_length` (95th pct) vào `config.yaml` | 🔴 P1 | Thi | `merged.csv`, `config.yaml` |
| 06/07 | Viết Cơ sở lý thuyết: kiến trúc + cơ chế huấn luyện (MLM, bỏ NSP, fine-tune) | 🟡 P2 | Ba Ka | Section `sn-article.tex` |
| **T3 07/07** | Stratified split 70/15/15 + tạo `seeds.json` (3 seed) | 🔴 P1 | Thi | `train/val/test.csv` 🔒 khóa từ đây |
| 07/07 | Code khung dùng chung: `dataset.py`, `train.py`, `evaluate.py` | 🔴 P1 | Ba Ka | `src/` chạy được |
| **T4 08/07** | B1 — Naïve Bayes + TF-IDF (`max_features=20000`, ngram(1,2)) | 🔴 P1 | Thi | `nb_tfidf.pkl` + log |
| 08/07 | B2 — SVM LinearSVC + TF-IDF (ngram(1,3)) | 🔴 P1 | Thi | `svm_tfidf.pkl` + log |
| 08/07 | Sửa metadata mẫu paper (title/author/abstract — bỏ đề tài Knee Osteoarthritis gốc) | 🟡 P2 | Bảo | Header `sn-article.tex` đúng |
| **T5 09/07** | Đọc + tóm tắt paper #9 (+ #10, #11 nếu kịp) | 🟡 P2 | Ba Ka | Đủ 9 tóm tắt |
| 09/07 | Bảng so sánh kiến trúc LSTM/GRU/BERT/PhoBERT | 🟢 P3 | Thi | Bảng trong paper |
| 09/07 | **Buffer** — bù task trễ trong tuần | — | Cả nhóm | — |

### 📅 TUẦN 2 (10/07 – 16/07): Deep Learning baseline + bắt đầu PhoBERT
> Gate cuối tuần: đủ 5 baseline, PhoBERT chạy run đầu tiên.

| Ngày | Task | Ưu tiên | Người | Đầu ra |
|---|---|---|---|---|
| **T6 10/07** | Tải FastText `cc.vi.300` + xây embedding layer | 🔴 P1 | Ba Ka | Embedding load được |
| **T7 11/07** | B3 — BiLSTM + FastText (train + tune nhẹ) | 🔴 P1 | Ba Ka | `bilstm_best.pt` + log |
| **CN 12/07** | B4 — GRU + FastText | 🔴 P1 | Bảo | `gru_best.pt` + log |
| **T2 13/07** | B5 — mBERT fine-tuned (`bert-base-multilingual-cased`) | 🔴 P1 | Bảo | `mbert_finetuned/` + log |
| 13/07 | Viết Methodology phần dữ liệu + tiền xử lý vào paper | 🟡 P2 | Thi | Section Methodology 50% |
| **T3 14/07** | PhoBERT run cơ sở (CLS head + class-weight) | 🔴 P1 | Ba Ka | `05_phobert_training.ipynb`, learning curve |
| **T4 15/07** | PhoBERT — thí nghiệm imbalance: class-weight vs Focal Loss | 🟡 P2 | Ba Ka | 2 run trong log |
| 15/07 | LLM đối chứng: viết script + prompt cố định gọi GPT/Gemini zero-shot | 🟡 P2 | Bảo | `07_llm_baseline.ipynb` khởi tạo |
| **T5 16/07** | PhoBERT — thí nghiệm head: CLS vs mean-pooling | 🟡 P2 | Ba Ka | 2 run trong log |
| 16/07 | **Buffer** — bù baseline trễ | — | Cả nhóm | — |

### 📅 TUẦN 3 (17/07 – 23/07): Tuning PhoBERT + LLM + Đánh giá toàn bộ
> Gate cuối tuần: chốt PhoBERT tốt nhất, chạy test toàn bộ, có bảng kết quả đầy đủ. Từ đây **không train lại nữa**.

| Ngày | Task | Ưu tiên | Người | Đầu ra |
|---|---|---|---|---|
| **T6 17/07** | PhoBERT — tuning (lr, batch, epoch) trên cấu hình tốt nhất | 🔴 P1 | Ba Ka | ≥5 dòng log, chọn checkpoint |
| **T7 18/07** | PhoBERT — chạy lại 3 seed → mean±std | 🟡 P2 | Ba Ka | `cv_results.csv` |
| 18/07 | LLM few-shot (thêm 3-6 ví dụ) + đo Acc/F1/NEU | 🟡 P2 | Bảo | `llm_predictions.csv` |
| **CN 19/07** | 🚪 GATE: nếu F1-macro PhoBERT < 0.70 → tăng tỉ trọng UIT-VSFC vào train, train lại | 🔴 P1 | Ba Ka | Quyết định + run bổ sung |
| **T2 20/07** | Chạy **test set** toàn bộ 7 model (Acc, P/R/F1 từng nhãn, NEU, time, params) | 🔴 P1 | Thi | Bảng kết quả đầy đủ |
| **T3 21/07** | Confusion matrix + AUC-ROC (OvR) mọi model | 🔴 P1 | Thi | `06_evaluation.ipynb` |
| 21/07 | McNemar test: PhoBERT vs từng baseline | 🟡 P2 | Thi | Bảng p-value |
| **T4 22/07** | Error analysis: phủ định, mỉa mai, câu ngắn, lẫn tiếng Anh — đếm + ví dụ cụ thể | 🟡 P2 | Cả nhóm | Bảng 5 loại lỗi trong paper |
| **T5 23/07** | Thí nghiệm attention-pooling head (chỉ nếu còn dư thời gian) | 🟢 P3 | Ba Ka | Run bổ sung |
| 23/07 | **Buffer** — bù đánh giá trễ | — | Cả nhóm | — |

### 📅 TUẦN 4 (24/07 – 01/08): Viết paper + Demo + Slide + Nộp
> Không làm thí nghiệm mới. Chỉ viết, tổng hợp, review.

| Ngày | Task | Ưu tiên | Người | Đầu ra |
|---|---|---|---|---|
| **T6 24/07** | Viết Numerical Results (bảng + biểu đồ + mean±std + McNemar) | 🔴 P1 | Ba Ka | Section Results xong |
| **T7 25/07** | Hoàn thiện Methodology (6 model + LLM + thí nghiệm head/loss) | 🔴 P1 | Thi | Section Methodology xong |
| **CN 26/07** | Viết Abstract + Introduction + đóng góp | 🔴 P1 | Bảo | 2 section xong |
| 26/07 | Demo Streamlit local (nhập câu → nhãn + confidence 3 lớp) | 🟡 P2 | Ba Ka | `app/app.py` chạy |
| **T2 27/07** | Viết Limitations/Discussion (từ error analysis) + Conclusion & Future Work | 🔴 P1 | Bảo | 2 section xong |
| 27/07 | Bibliography ≥15 nguồn `\bibitem` | 🟡 P2 | Cả nhóm | thebibliography đầy đủ |
| **T3 28/07** | Điền slide Beamer `Mau_baocao.tex` (~12–15 slide, xem mục 7) | 🔴 P1 | Bảo | Slide xong |
| 28/07 | Deploy demo HuggingFace Spaces (nếu kịp) | 🟢 P3 | Ba Ka | Link công khai |
| **T4 29/07** | Review chéo toàn bộ paper — soát số liệu, hình, trích dẫn | 🔴 P1 | Cả nhóm | Bản nháp hoàn chỉnh |
| **T5 30/07** | Sửa theo review + compile PDF + `README.md` + push final | 🔴 P1 | Cả nhóm | Paper PDF hoàn chỉnh |
| **T6 31/07** | **Buffer lớn** — dự phòng mọi thứ trễ | 🔴 P1 | Cả nhóm | — |
| **T7 01/08** | Duyệt lần cuối + **NỘP** | 🔴 P1 | Cả nhóm | Paper + slide + demo |

---

## 7. BẢN ĐỒ NỘI DUNG → `sn-article.tex`

| Section trong mẫu | Nội dung điền | Viết ở ngày |
|---|---|---|
| `Introduction` | Đặt vấn đề SA tiếng Việt, mục tiêu, đóng góp (research question mục 1) | 26/07 |
| `Fundamental Definitions` | Cơ sở lý thuyết + cơ chế huấn luyện (từ Tuần 1) | 06/07 |
| `Methodology` | Dữ liệu (gốc + UIT-VSFC), tiền xử lý, 6 kiến trúc, thí nghiệm head & loss | 13/07 → 25/07 |
| `Numerical Results` | Bảng so sánh (mục 8) + mean±std + McNemar + biểu đồ | 24/07 |
| `Limitations and Discussions` | Bảng error analysis 5 loại lỗi | 27/07 |
| `Conclusion And Future Work` | Kết luận + hướng phát triển cụ thể | 27/07 |

**Cấu trúc slide `Mau_baocao.tex`** (28/07): Giới thiệu bài toán → Dữ liệu → 6 mô hình → Thí nghiệm head & loss (đóng góp) → Bảng kết quả + significance → Demo → Kết luận.

---

## 8. BẢNG KẾT QUẢ MẪU (điền dần khi có số liệu thật, chốt ở 20/07)

| # | Mô hình | Loại | Acc | F1-macro (mean±std) | F1-Tiêu cực | NEU (Bình thường) | Time (ms) | Params |
|---|---|---|---|---|---|---|---|---|
| B1 | NB + TF-IDF | ML | — | — | — | — | — | — |
| B2 | SVM + TF-IDF | ML | — | — | — | — | — | — |
| B3 | BiLSTM + FastText | DL | — | — | — | — | — | ~3M |
| B4 | GRU + FastText | DL | — | — | — | — | — | ~2.5M |
| B5 | mBERT fine-tuned | PLM | — | — | — | — | — | ~178M |
| L | LLM (GPT/Gemini, few-shot) | LLM | — | — | — | — | — | API |
| **P** | **PhoBERT (best config)** | **PLM** | **—** | **—** | **—** | **—** | **—** | **~135M** |

---

## 9. RỦI RO & PHƯƠNG ÁN DỰ PHÒNG

| Rủi ro | Mức | Xử lý | Ngày kiểm tra |
|---|---|---|---|
| Nhãn Tiêu cực quá ít (~7%) | 🔴 Cao | Gộp UIT-VSFC ngay Tuần 1 + class-weight/Focal Loss | 06/07 |
| Data nhỏ → PhoBERT overfit | 🔴 Cao | Early stopping + dropout 0.3 + weight decay; đánh giá bằng 3-seed thay vì 1 lần | 18/07 |
| Chênh PhoBERT vs baseline không có ý nghĩa thống kê | 🟡 TB | McNemar test; nếu p>0.05 → thêm data hoặc thử head khác | 21/07 |
| Thiếu GPU / hết quota Colab | 🟡 TB | Giảm `max_len=128`, `batch=8`, dùng 1 seed thay 3 | Bất kỳ lúc nào |
| Trễ tiến độ Tuần 2–3 (giai đoạn nặng nhất) | 🟡 TB | Dùng ngày buffer 16/07, 23/07; cắt thẳng task 🟢 P3 | Cuối mỗi tuần |
| GV không phản hồi định nghĩa NEU score kịp | 🟢 Thấp | Tự chọn F1 lớp Neutral làm mặc định, ghi rõ giả định trong paper | 03/07 |

---

## 10. PHÂN CÔNG TỔNG HỢP THEO LUỒNG

| Luồng | Người | Tuần 1 | Tuần 2 | Tuần 3 | Tuần 4 |
|---|---|---|---|---|---|
| **DATA** | Thái Ngọc Thi | EDA, gộp data, split, NB, SVM | Methodology (data) | Test toàn bộ, confusion, McNemar | Viết Methodology hoàn chỉnh |
| **MODEL** | Mô Ha Mách Bu Ba Ka | Lý thuyết, khung code | BiLSTM, PhoBERT setup, head/loss | Tuning, 3-seed, gate quyết định | Numerical Results, demo |
| **HẠ TẦNG + VIẾT** | Lê Đình Bảo | Repo, tóm tắt paper Việt, metadata | GRU, mBERT, LLM script | LLM few-shot, error analysis | Abstract/Intro/Conclusion, slide, README |

---

## 11. CHECKLIST "HOÀN THÀNH TỐI THIỂU" (chỉ cần P1 là nộp được)

- [ ] Repo đúng cấu trúc mục 2, có README
- [ ] Dữ liệu đã gộp UIT-VSFC, split 70/15/15, test khóa từ 07/07
- [ ] 5 baseline (NB, SVM, BiLSTM, GRU, mBERT) có kết quả val + test
- [ ] PhoBERT fine-tune xong, có ít nhất 1 cấu hình tốt nhất
- [ ] Bảng so sánh Acc/F1-macro/NEU/time cho toàn bộ 6–7 model
- [ ] `sn-article.tex` đủ 6 section, ≥15 tài liệu tham khảo, metadata đã sửa đúng đề tài
- [ ] `Mau_baocao.tex` slide 12–15 trang
- [ ] Demo chạy được (local là đủ)

**Nếu kịp thêm (P2 nâng chất lượng):** đối chứng LLM, McNemar test, Focal Loss, mean±std 3-seed, deploy demo public.

---

*Kế hoạch 1 tháng — bản đầy đủ nhất, gộp lịch ngày + cấu trúc thư mục + tech stack + paper list + rủi ro | 03/07 → 01/08/2026 | Dùng kèm `HUONG_DAN_CHI_TIET_TASK.md` để tra format từng loại task*
