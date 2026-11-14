# Quy trình thực nghiệm và xử lý dữ liệu

## A. Quy trình xử lý ảnh trên giao diện

1. Chọn ảnh đầu vào bằng nút **Mở ảnh** (ảnh đang mở được dùng làm ảnh gốc / ground truth).
2. Chọn loại nhiễu (Gaussian hoặc Salt & Pepper) hoặc giữ nguyên không thêm nhiễu.
3. Đặt mức nhiễu, thông số Gamma / Tương phản / Độ sáng, bộ lọc (Mean hoặc Median) và kích thước kernel.
4. Nhấn **Áp dụng xử lý**. Các chỉ số MSE/PSNR/SSIM/Edge F1 được tự động tính so với ảnh gốc (đã áp dụng cùng biến đổi Gamma/Contrast/Brightness).
5. Chuyển sang tab **Thực nghiệm** và nhấn **Chạy 72 ca thực nghiệm** để tự động khảo sát toàn diện.

## B. Ma trận thực nghiệm tự động

- Nhiễu: Gaussian sigma 12, 24; Salt & Pepper 3%, 7%.
- Gamma: 0.8, 1.0, 1.2.
- Bộ lọc: Mean, Median.
- Kernel: 3×3, 5×5, 7×7.
- Tổng số cấu hình: 4 × 3 × 6 = 72.
- Seed sinh nhiễu cố định 2026 để tái lập.
- Kết quả: CSV, biểu đồ PSNR và Edge F1, ảnh minh họa.

## C. Diễn giải metric

- MSE và PSNR tính trên toàn bộ kênh BGR; MAX=255.
- SSIM cục bộ tính trên ảnh xám, cửa sổ 7×7.
- Edge F1 dùng Sobel và một ngưỡng chung bằng 15% cực đại gradient của ảnh tham chiếu.
- Gamma được áp dụng lên cả ảnh tham chiếu và ảnh xử lý trước khi so sánh. Điều này đo sai số khử nhiễu trong cùng miền cường độ.
