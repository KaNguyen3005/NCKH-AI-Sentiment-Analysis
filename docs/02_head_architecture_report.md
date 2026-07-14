# Báo cáo So sánh Kiến trúc Head (CLS vs Mean-Pooling)

Thí nghiệm tập trung kiểm định sự khác biệt về mặt trích xuất đặc trưng khi thay đổi cấu trúc ở các lớp cuối cùng của PhoBERT trước khi đưa vào hàm phân loại. Cả hai thực nghiệm đều được cố định bằng **Focal Loss** nhằm giữ tính công bằng và đối chiếu xem phương pháp biểu diễn câu nào tốt hơn.

## 1. Kết quả thực nghiệm

Kết quả được thu thập từ file `results/experiments_log.csv`:

| Kiến trúc Head | Accuracy | F1-Macro | F1 - Tiêu cực | **F1 - Bình thường (Thiểu số)** | F1 - Tích cực |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`[CLS]` Token** | **0.9007** | **0.8064** | **0.9400** | **0.5550** | **0.9242** |
| **Mean-Pooling** | 0.8425 | 0.7452 | 0.9251 | 0.4386 | 0.8719 |

## 2. Phân tích kết quả

> [!WARNING]
> Kỹ thuật **Mean-Pooling** chứng kiến sự sụt giảm nghiêm trọng trên tất cả các thang đo so với việc sử dụng `[CLS]` token truyền thống.

- Đối với các mô hình ngôn ngữ họ RoBERTa (như PhoBERT), token `<s>` (đóng vai trò như `[CLS]`) đã được thiết kế sẵn và tối ưu hóa cực kỳ tốt thông qua quá trình pre-training khổng lồ để đại diện cho toàn bộ ngữ nghĩa của câu (sentence-level representation).
- Khi ta cố tình gạt bỏ vector `[CLS]` này và tự cào bằng bằng cách lấy trung bình (Mean-Pooling) tất cả các subwords, mô hình mất đi tính tập trung ngữ nghĩa cốt lõi. Trong một tập dữ liệu phân tích cảm xúc (vốn chứa nhiều từ trung tính đan xen một vài từ khóa cảm xúc mạnh), việc cào bằng khiến tín hiệu từ các từ khóa bị pha loãng.
- Sự pha loãng này thể hiện rất rõ ở nhãn thiểu số (nhãn `Bình thường`), khi mà F1-score rớt thê thảm từ **55.5%** xuống chỉ còn **43.86%**. Accuracy chung cũng giảm từ **~90%** xuống **~84%**.

## 3. Kết luận và Đề xuất

> [!TIP]
> Trả lại kiến trúc **`[CLS]` Head mặc định**. Không nên sử dụng Mean-Pooling cho bài toán phân loại sắc thái tình cảm đối với mô hình PhoBERT.

Dù kiến trúc `CustomPhoBERTClassifier` đã cho phép cấu hình linh hoạt, nhưng thực tế chứng minh mô hình hoạt động hiệu quả nhất với thiết lập nguyên bản của nó. Bạn có thể sử dụng lại trọng số ở file `best_model_focal_cls.pt` cho các bước Inference (dự đoán) tiếp theo trên ứng dụng của mình.
