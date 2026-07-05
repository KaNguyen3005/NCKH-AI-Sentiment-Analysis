# 📖 HƯỚNG DẪN CHI TIẾT TỪNG LOẠI TASK
## Tài liệu bổ sung cho `KE_HOACH_1_THANG.md` — "làm như thế nào", không chỉ "làm gì"

> Dùng file này song song với kế hoạch 30 ngày. Tới task nào, tra mục tương ứng ở đây trước khi bắt tay.

---

## 1️⃣ ĐỌC & TÓM TẮT PAPER

**Vấn đề nếu không có chuẩn:** mỗi người tóm tắt một kiểu, không dùng lại được khi viết Related Works.

### Format bắt buộc — mỗi paper là 1 khối trong `docs/related_works.md`:

```markdown
## [3] Devlin et al. (2019) — BERT: Pre-training of Deep Bidirectional Transformers
- **Nguồn:** arxiv.org/abs/1810.04805 (NAACL 2019)
- **Bài toán họ giải:** Học biểu diễn ngôn ngữ tổng quát bằng pre-training 2 chiều.
- **Phương pháp cốt lõi:** Masked Language Model (che 15% token, dự đoán lại) +
  Next Sentence Prediction, kiến trúc Transformer encoder 12/24 lớp.
- **Kết quả chính:** SOTA trên 11 tác vụ NLP tại thời điểm công bố (GLUE, SQuAD).
- **Liên quan đề tài của nhóm:** PhoBERT (mô hình chính) dùng cùng cơ chế MLM
  nhưng bỏ NSP, huấn luyện lại từ đầu trên corpus tiếng Việt.
- **1 câu trích dẫn được phép dùng trong paper (nếu cần):** không quote nguyên văn,
  chỉ diễn giải lại bằng lời của nhóm.
```

### Quy tắc 5 dòng bắt buộc (đúng 5 gạch đầu dòng, không thêm không bớt):
1. **Nguồn** — link + venue/năm
2. **Bài toán họ giải** — 1 câu
3. **Phương pháp cốt lõi** — kiến trúc/kỹ thuật chính, 1–2 câu
4. **Kết quả chính** — con số cụ thể nếu có
5. **Liên quan đề tài của nhóm** — bắt buộc phải nối được vào bài của nhóm, không viết chung chung

### ⚠️ Quy tắc chống đạo văn/vi phạm bản quyền khi viết vào paper
- **Không copy-paste câu tiếng Anh** rồi dịch máy — diễn giải lại bằng hiểu biết của mình.
- Khi trích dẫn trong `sn-article.tex`, dùng `\cite{}` trỏ tới `.bib`, **không chép nguyên đoạn**.
- Mỗi paper chỉ nên xuất hiện 1 lần trong phần Related Works, đừng lặp lại ở nhiều chỗ.

### Ai đọc bài nào (đã chia ở kế hoạch) → chỉ cần điền đúng khuôn trên, mất ~20–25 phút/bài.

---

## 2️⃣ EDA (Exploratory Data Analysis) — `01_EDA.ipynb`

**Không phải "vẽ vài biểu đồ cho có" — phải trả lời được 5 câu hỏi cụ thể, mỗi câu 1 code cell + 1 nhận xét bằng chữ:**

| # | Câu hỏi | Code cần chạy | Vì sao cần |
|---|---|---|---|
| 1 | Phân phối nhãn thế nào? Mất cân bằng bao nhiêu %? | `df['label'].value_counts(normalize=True)` + bar chart | Quyết định có cần class-weight/Focal Loss không |
| 2 | Độ dài câu (số từ) phân phối ra sao? | Histogram + `df['text'].str.split().str.len().describe()` | Chốt `max_length` (dùng percentile 95) |
| 3 | Có câu trùng lặp / gần trùng không? | `df.duplicated().sum()` + kiểm tra fuzzy nếu cần | Trùng lặp giữa train/test sẽ làm sai lệch đánh giá |
| 4 | Từ nào xuất hiện nhiều nhất mỗi nhãn? | Wordcloud hoặc bar chart top-20 từ/nhãn (sau khi bỏ stopword) | Phát hiện từ khóa đặc trưng, phát hiện nhiễu (link, tên riêng) |
| 5 | Có câu lẫn ngôn ngữ khác (tiếng Anh, teencode, emoji) không? | Regex đếm số câu chứa ký tự Latin không dấu bất thường / emoji | Biết cần xử lý gì ở bước tiền xử lý |

**Định dạng output bắt buộc cuối notebook — 1 bảng tóm tắt:**
```
Tổng số mẫu: ____
Phân phối nhãn: Tích cực __%  Bình thường __%  Tiêu cực __%
Độ dài trung bình: __ từ | Percentile 95: __ từ → chọn max_length = __
Số câu trùng: __
Nhận xét rủi ro chính: (vd "nhãn Tiêu cực chỉ 7%, cần xử lý imbalance")
```
Bảng này copy thẳng vào phần Methodology của paper.

---

## 3️⃣ TIỀN XỬ LÝ — `src/preprocessing.py`

**Hàm bắt buộc phải có, đúng thứ tự xử lý (thứ tự sai sẽ ra kết quả khác):**

```python
def clean_text(text: str) -> str:
    """
    Thứ tự xử lý CỐ ĐỊNH:
    1. Chuẩn hóa Unicode NFC (tiếng Việt hay bị 2 dạng encode khác nhau)
    2. Lowercase
    3. Xóa URL, email
    4. Xóa HTML tag nếu có
    5. Chuẩn hóa emoji (giữ lại nhưng convert về dạng thống nhất, hoặc xóa tùy quyết định)
    6. Chuẩn hóa khoảng trắng (nhiều space -> 1 space, strip)
    7. KHÔNG xóa dấu câu hoàn toàn (dấu ? ! quan trọng cho sentiment)
    """
```

**Bắt buộc có unit test ngay trong file** (không phải optional):
```python
# Test 3 câu mẫu — chạy python src/preprocessing.py để tự kiểm tra
test_cases = [
    "Cái app này dở òm 😡😡 xem  http://link.com đi",
    "SẢN PHẨM tốt&nbsp;quá!!! Recommend luôn",
    "bình thường thôi, không có gì đặc biệt...",
]
for t in test_cases:
    print(f"{t!r} -> {clean_text(t)!r}")
```

**2 nhánh xử lý khác nhau (ghi rõ trong file, đừng lẫn lộn):**
- Nhánh **ML/RNN** (NB, SVM, BiLSTM, GRU): cần tách từ bằng `underthesea.word_tokenize()` sau `clean_text()`.
- Nhánh **PLM** (mBERT, PhoBERT): dùng tokenizer riêng của model (`AutoTokenizer.from_pretrained(...)`), **không** tách từ underthesea trước — PhoBERT có tokenizer BPE riêng.

---

## 4️⃣ HUẤN LUYỆN MODEL — quy tắc chung cho mọi baseline

**Mỗi lần train xong 1 model, bắt buộc ghi 1 dòng vào `experiments_log.csv` với đúng các cột sau — không được thiếu cột nào:**

```csv
date,model,config,train_size,val_acc,val_f1_macro,val_f1_negative,val_f1_neutral,val_f1_positive,train_time_min,notes
2026-07-08,SVM_TFIDF,"ngram(1,3) C=1.0",700,0.81,0.72,0.55,0.78,0.85,2,"baseline nhanh"
```

**Quy tắc đặt tên file model** (để không lẫn lộn khi có nhiều run):
```
models/baseline/{tên_model}_{ngày}_{version}.{pkl|pt}
vd: svm_tfidf_0708_v1.pkl, phobert_0715_focalloss_v2/
```

**Với mô hình sâu (BiLSTM/GRU/mBERT/PhoBERT), bắt buộc lưu learning curve:**
- Vẽ train loss + val loss theo epoch trên cùng 1 hình → lưu vào `figures/{model}_learning_curve.png`
- Nếu val loss tăng trở lại (overfit) → ghi rõ epoch dừng vào log.

**Split dữ liệu — nhắc lại quy tắc cứng:**
- Train dùng để học tham số.
- Val dùng để chọn model/tune hyperparameter — **được xem nhiều lần**.
- Test **chỉ chạy 1 lần duy nhất, ở Tuần 3, sau khi đã chốt xong mọi cấu hình**.

---

## 5️⃣ ĐÁNH GIÁ MODEL — `07_evaluation.ipynb`

**Với mỗi model, bắt buộc tính đủ 6 chỉ số này (không được chỉ báo Accuracy):**

| Chỉ số | Cách tính | Vì sao cần |
|---|---|---|
| Accuracy | `accuracy_score` | Chỉ số tổng quát, dễ hiểu nhưng gây hiểu lầm khi imbalance |
| F1-macro | `f1_score(average='macro')` | Chỉ số chính của đề tài — coi trọng nhãn thiểu số ngang nhãn đa số |
| F1 từng nhãn (Tích cực/Bình thường/Tiêu cực) | `classification_report` | Biết model yếu ở nhãn nào |
| **NEU score** | F1 riêng của nhãn Bình thường (đã chốt định nghĩa ở 03/07) | Yêu cầu riêng của đề tài |
| Inference time | Đo thời gian dự đoán 100 câu, chia trung bình, tính bằng ms/câu, đo cả CPU và GPU nếu có | Đề tài yêu cầu "tốc độ xử lý" |
| Params | `sum(p.numel() for p in model.parameters())` (model sâu) hoặc số feature (ML) | So sánh độ phức tạp |

**Confusion matrix** — vẽ dạng heatmap (`seaborn.heatmap`), chuẩn hóa theo hàng (%) để thấy model nhầm nhãn nào sang nhãn nào, không chỉ số tuyệt đối.

**McNemar test — cách làm cụ thể:**
```python
from statsmodels.stats.contingency_tables import mcnemar
# Bảng 2x2: model A đúng/sai vs model B đúng/sai trên CÙNG tập test
# contingency_table[0][0] = cả 2 đúng
# contingency_table[0][1] = A đúng, B sai
# contingency_table[1][0] = A sai, B đúng
# contingency_table[1][1] = cả 2 sai
result = mcnemar(contingency_table, exact=True)
# p < 0.05 => chênh lệch có ý nghĩa thống kê, không phải do may rủi
```

---

## 6️⃣ ERROR ANALYSIS — định dạng cụ thể

**Không viết chung chung "model hay nhầm câu phủ định". Phải có bảng cụ thể:**

| Loại lỗi | Số câu | % trong tổng lỗi | Ví dụ cụ thể (câu + nhãn thật + nhãn dự đoán) |
|---|---|---|---|
| Câu phủ định ("không tệ", "chẳng có gì hay") | __ | __% | "Sản phẩm không tệ" → thật: Tích cực, dự đoán: Tiêu cực |
| Câu mỉa mai/châm biếm | __ | __% | ... |
| Câu quá ngắn (<5 từ) | __ | __% | ... |
| Lẫn tiếng Anh/teencode | __ | __% | ... |
| Nhãn Bình thường bị nhầm sang 2 cực | __ | __% | ... |

**Cách lấy dữ liệu:** lọc `y_true != y_pred` trên tập test, đọc thủ công 30–50 câu sai của model tốt nhất (PhoBERT), tự phân loại vào 5 nhóm lỗi trên bằng mắt.

---

## 7️⃣ LLM ĐỐI CHỨNG (zero-shot & few-shot)

**Prompt chuẩn zero-shot (dùng cố định cho mọi câu, không đổi giữa chừng để đảm bảo công bằng):**
```
Bạn là công cụ phân loại cảm xúc. Đọc câu tiếng Việt sau và trả lời
CHỈ MỘT trong 3 nhãn: "Tích cực", "Bình thường", "Tiêu cực".
Không giải thích, chỉ trả về đúng 1 nhãn.

Câu: "{text}"
Nhãn:
```

**Prompt few-shot** — thêm 3–6 ví dụ mẫu (2 mỗi nhãn) lấy từ tập **train** (không lấy từ test) trước câu cần phân loại.

**Cách chạy:** loop qua tập test, gọi API, parse câu trả lời về đúng 1 trong 3 nhãn (nếu model trả lời sai định dạng → tính là dự đoán sai). Ghi lại toàn bộ input/output vào `results/llm_predictions.csv` để tái kiểm tra.

---

## 8️⃣ VIẾT PAPER (`sn-article.tex`) — quy tắc mỗi section

| Section | Độ dài mục tiêu | Nội dung bắt buộc phải có |
|---|---|---|
| Abstract | 150–250 từ | Bài toán → phương pháp → kết quả số cụ thể → đóng góp, viết SAU CÙNG khi đã có số liệu |
| Introduction | 1–1.5 trang | Bối cảnh, vấn đề, mục tiêu, liệt kê 3 đóng góp bằng gạch đầu dòng |
| Fundamental Definitions | 2–3 trang | Từ `related_works.md`, viết lại bằng lời riêng, có sơ đồ kiến trúc |
| Methodology | 3–4 trang | Dữ liệu (bảng thống kê từ EDA) → tiền xử lý → kiến trúc 6+1 model → thí nghiệm head/loss |
| Numerical Results | 2–3 trang | Bảng so sánh đầy đủ + biểu đồ + McNemar, KHÔNG diễn giải dài dòng, để số liệu tự nói |
| Limitations and Discussions | 1–2 trang | Từ bảng error analysis ở mục 6 phía trên |
| Conclusion | 0.5–1 trang | Tóm tắt đóng góp + hướng phát triển cụ thể (không chung chung "sẽ cải thiện thêm") |

**Quy tắc viết số liệu:** mọi con số trong Results phải trỏ được về đúng 1 dòng trong `experiments_log.csv` hoặc `cv_results.csv` — không tự ước lượng, không làm tròn tùy tiện.

---

## 9️⃣ SLIDE (`Mau_baocao.tex`) — 1 slide = 1 ý

Quy tắc cứng: **mỗi slide tối đa 5 gạch đầu dòng, mỗi dòng tối đa 15 từ.** Bảng/biểu đồ chiếm slide riêng, không nhét chung với text dài.

Khung 13 slide gợi ý:
1. Trang bìa | 2. Đặt vấn đề | 3. Mục tiêu | 4. Dữ liệu (bảng thống kê) | 5. Tiền xử lý | 6–7. Kiến trúc các model (2 slide) | 8. Thí nghiệm head/loss | 9. Bảng kết quả chính | 10. Biểu đồ so sánh | 11. Error analysis | 12. Demo (ảnh chụp) | 13. Kết luận + hướng phát triển.

---

## 🔟 DEMO STREAMLIT — checklist tối thiểu

```python
# app/app.py — checklist chức năng bắt buộc
# [ ] Ô nhập text tiếng Việt
# [ ] Nút "Phân tích" 
# [ ] Hiển thị nhãn dự đoán + % confidence 3 lớp (bar chart)
# [ ] Dùng model PhoBERT tốt nhất đã lưu ở models/phobert/best_checkpoint/
# [ ] (P3 nếu kịp) So sánh song song dự đoán của SVM vs PhoBERT
```

---

## 📌 TÓM TẮT — CHECKLIST FORMAT NHANH

| Task | File | Format bắt buộc |
|---|---|---|
| Tóm tắt paper | `related_works.md` | 5 gạch đầu dòng cố định/paper |
| EDA | `01_EDA.ipynb` | 5 câu hỏi + bảng tóm tắt cuối |
| Tiền xử lý | `preprocessing.py` | Đúng 7 bước thứ tự + unit test |
| Log thí nghiệm | `experiments_log.csv` | Đúng 11 cột, 1 dòng/lần train |
| Đánh giá | `07_evaluation.ipynb` | 6 chỉ số + confusion matrix + McNemar |
| Error analysis | Bảng 5 loại lỗi | Có ví dụ câu cụ thể, không chung chung |
| LLM | `llm_predictions.csv` | Prompt cố định, lưu toàn bộ input/output |
| Paper | `sn-article.tex` | Đúng độ dài từng section |
| Slide | `Mau_baocao.tex` | Max 5 bullet/slide, 15 từ/dòng |

---

*Sổ tay chi tiết — dùng kèm `KE_HOACH_1_THANG.md`. Ai làm task nào, tra đúng mục ở đây trước khi bắt tay.*
