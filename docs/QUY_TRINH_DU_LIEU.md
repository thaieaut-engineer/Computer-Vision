# Quy trình sử dụng hai bộ dữ liệu

## A. Bộ dữ liệu ảnh sạch (thực nghiệm có kiểm soát)

1. Chọn ảnh sạch làm ảnh đầu vào bằng **Mở ảnh**.
2. Không cần chọn ảnh tham chiếu riêng; ảnh đang mở sẽ được xem là ground truth.
3. Chọn Gaussian hoặc Salt & Pepper, đặt mức nhiễu, Gamma 0.8/1.0/1.2, Mean/Median và kernel.
4. Nhấn **Áp dụng xử lý**. MSE/PSNR/SSIM/Edge F1 được tính so với ảnh sạch đã áp dụng cùng biến đổi Gamma/Contrast/Brightness.
5. Với tab **Thực nghiệm**, chạy 72 cấu hình. Nếu không chọn ảnh tham chiếu riêng, ứng dụng hỏi xác nhận ảnh đang mở có sạch hay không.

## B. Bộ dữ liệu ảnh thực tế đã nhiễu

1. Chọn ảnh đã nhiễu làm ảnh đầu vào bằng **Mở ảnh**.
2. Trong nhóm **Ảnh tham chiếu sạch (Ground Truth)**, chọn ảnh sạch của đúng cảnh đó.
3. Đảm bảo ảnh sạch và ảnh nhiễu đã căn chỉnh cùng vị trí, cùng kích thước, cùng số kênh. Chỉ cùng kích thước nhưng khác cảnh/góc chụp không phải ground truth hợp lệ.
4. Đặt **Loại nhiễu = Không thêm nhiễu** để không chồng thêm nhiễu nhân tạo.
5. Chọn Mean/Median và kernel, nhấn **Áp dụng xử lý**. Các chỉ số được tính với ảnh tham chiếu sạch sau cùng biến đổi điểm ảnh.
6. Nếu không có ảnh sạch tương ứng, không thể báo cáo MSE/PSNR/SSIM/Edge F1 như các chỉ số khôi phục tham chiếu. Ứng dụng sẽ hiển thị N/A. Khi đó chỉ phân tích định tính, histogram và quan sát trực quan.

## C. Ma trận thực nghiệm tự động

- Nhiễu: Gaussian sigma 12, 24; Salt & Pepper 3%, 7%.
- Gamma: 0.8, 1.0, 1.2.
- Bộ lọc: Mean, Median.
- Kernel: 3×3, 5×5, 7×7.
- Tổng số cấu hình: 4 × 3 × 6 = 72.
- Seed sinh nhiễu cố định 2026 để tái lập.
- Kết quả: CSV, biểu đồ PSNR và Edge F1, ảnh minh họa.

## D. Diễn giải metric

- MSE và PSNR tính trên toàn bộ kênh BGR; MAX=255.
- SSIM cục bộ tính trên ảnh xám, cửa sổ 7×7.
- Edge F1 dùng Sobel và một ngưỡng chung bằng 15% cực đại gradient của ảnh tham chiếu.
- Gamma được áp dụng lên cả ảnh tham chiếu và ảnh xử lý trước khi so sánh. Điều này đo sai số khử nhiễu trong cùng miền cường độ.
