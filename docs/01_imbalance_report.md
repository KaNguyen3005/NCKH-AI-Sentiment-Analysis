# Báo cáo so sánh kỹ thuật xử lý Imbalance Data cho PhoBERT

Sau khi hoàn tất quá trình huấn luyện (fine-tuning) mô hình `vinai/phobert-base` với 3 phương pháp hàm Loss khác nhau, dưới đây là kết quả đánh giá trên tập Validation:

## 1. Kết quả thực nghiệm

Kết quả được ghi nhận lại từ log hệ thống (`results/experiments_log.csv`):

| Phương pháp | Accuracy (Tổng quát) | F1-Macro (Trung bình 3 nhãn) | F1 - Tiêu cực | **F1 - Bình thường (Thiểu số)** | F1 - Tích cực |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cross-Entropy (CE)** | **0.9189** | 0.7899 | 0.9440 | **0.4895** | 0.9362 |
| **Class Weight** | 0.9011 | 0.8079 | 0.9335 | **0.5676** | 0.9227 |
| **Focal Loss** | 0.9063 | **0.8169** | 0.9423 | **0.5805** | 0.9280 |

*(Tất cả đều chạy 4 epochs, learning rate = 2e-5, batch_size = 32).*

## 2. Phân tích kết quả

> [!IMPORTANT]
> Nhãn `Bình thường` là nhãn chịu mất cân bằng nghiêm trọng nhất (chỉ chiếm ~5.78% dữ liệu train). Do đó, chỉ số **F1 - Bình thường** là mục tiêu so sánh quan trọng nhất trong thí nghiệm này.

- **Baseline (Cross-Entropy thuần):** Mô hình có xu hướng học theo nhãn đa số. Dù đạt Accuracy tổng quát cao nhất (91.89%) nhưng F1-score của nhãn Bình thường rất thấp, chỉ đạt khoảng **48.95%**.
- **Cross-Entropy với Class Weight:** Việc tính toán trọng số động và bù đắp lỗi phạt cho nhóm thiểu số đã khiến hiệu năng nhận diện nhãn Bình thường tăng vọt lên **56.76%** (tăng hơn ~8%). Kéo theo đó, Accuracy tổng quát giảm nhẹ xuống khoảng 90.11% do mô hình không còn thiên vị quá mức nhãn đa số.
- **Focal Loss ($\gamma=2.0$):** Là phương pháp **vượt trội nhất**. Thuật toán không chỉ tập trung vào phân bố mẫu mà còn đánh phạt những mẫu khó học. Kết quả ghi nhận F1 nhãn Bình thường đạt mức cao nhất **58.05%**, đồng thời kéo điểm F1-Macro chung (trung bình 3 nhãn) lên mức **81.69%**.

## 3. Kết luận và Khuyến nghị

> [!TIP]
> **Khuyến nghị sử dụng Focal Loss** cho các bước triển khai tiếp theo.

Phương pháp **Focal Loss** mang lại độ cân bằng tốt nhất: cải thiện mạnh mẽ khả năng phân loại nhãn thiểu số (Bình thường) mà vẫn bảo toàn điểm số rất tốt ở các nhãn đa số (đạt 94% ở Tiêu cực và gần 93% ở Tích cực). 

### File trọng số lưu trữ
Các trọng số từ tiến trình cũng đã được tự động lưu theo định dạng đặt tên để bạn tiện tái sử dụng trong tương lai tại mục `models/phobert/best_checkpoint/`:
- `best_model_ce.pt`
- `best_model_class_weight.pt`
- `best_model_focal.pt`
