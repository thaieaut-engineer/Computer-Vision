# Cơ sở thuật toán và cách tự cài đặt

Tài liệu này mô tả phần thuật toán do nhóm tự viết. NumPy chỉ được dùng làm cấu trúc lưu trữ ma trận, chuyển kiểu, cấp phát mảng, đệm biên và sinh số ngẫu nhiên; không gọi các hàm xử lý ảnh dựng sẵn như `cv2.blur`, `cv2.medianBlur`, `cv2.Sobel` hoặc hàm SSIM/MSE có sẵn.

## 1. Gamma Correction

Công thức: `s = 255 * (r / 255)^gamma`. Mỗi pixel được biến đổi theo công thức, làm tròn và giới hạn trong [0,255]. Gamma < 1 thường nâng sáng vùng tối/trung gian; gamma > 1 thường làm tối.

## 2. Điều chỉnh tương phản và độ sáng

`g(x,y) = alpha * f(x,y) + beta`. Alpha điều khiển độ tương phản, beta điều khiển độ sáng. Kết quả được làm tròn và cắt về [0,255].

## 3. Mean Filter – tự tính tổng vùng lân cận

Với kernel kích thước `k × k`, thuật toán thực hiện cho từng pixel:

1. Đọc `k²` pixel trong vùng lân cận.
2. Cộng từng vùng pixel tương ứng với từng vị trí kernel bằng hai vòng lặp qua kernel; phép cộng trên các vùng mảng giúp tăng tốc, không gọi hàm mean/box filter.
3. Chia tổng cho `k²`, làm tròn và gán vào pixel đầu ra.

Không dùng `np.mean`, `sliding_window_view`, `cv2.blur` hoặc `cv2.boxFilter`. Biên ảnh được mở rộng bằng `reflect`. Độ phức tạp xấp xỉ `O(H × W × k²)` cho mỗi kênh. Mean giảm nhiễu ngẫu nhiên nhưng có thể làm mờ cạnh.

## 4. Median Filter – tự chọn hạng bằng tìm kiếm nhị phân

Với cửa sổ có `k²` pixel, trung vị là phần tử ở hạng `k² // 2` sau khi sắp xếp tăng dần. Để tăng tốc trên ảnh lớn, chương trình tự cài đặt thuật toán chọn hạng:

1. Xét miền giá trị pixel 8-bit từ 0 đến 255.
2. Chọn giá trị giữa `mid` của miền đang xét.
3. Đếm số pixel trong cửa sổ có giá trị `<= mid`.
4. Nếu số lượng lớn hơn hạng trung vị, thu hẹp miền về nửa thấp; ngược lại thu hẹp về nửa cao.
5. Lặp đến khi cận dưới bằng cận trên; đó là giá trị trung vị.

Thuật toán cần tối đa 8 vòng tìm kiếm cho ảnh uint8. Không dùng `np.median`, `np.partition`, `np.sort`, `cv2.medianBlur` hoặc hàm median dựng sẵn. Hàm `insertion_sort` cũng được cung cấp trong module để minh họa cách sắp xếp cơ bản và kiểm thử, nhưng đường chạy chính dùng rank selection để tăng tốc. Median hiệu quả với nhiễu muối tiêu và thường giữ cạnh tốt hơn Mean.

## 5. Sobel – tích chập 3×3 tự cài đặt

Hai kernel:

`Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]`

`Gy = [[-1,-2,-1],[0,0,0],[1,2,1]]`

Thuật toán duyệt 9 vị trí của kernel, nhân hệ số kernel với vùng ảnh được dịch tương ứng rồi cộng dồn để thu được Gx và Gy. Các phép nhân-cộng trên mảng giúp tăng tốc, nhưng kernel Sobel và phép tích chập được triển khai tường minh; không gọi hàm Sobel/filter2D dựng sẵn. Độ lớn gradient được tính bằng `sqrt(Gx² + Gy²)`. Không dùng `np.einsum`, `sliding_window_view`, `cv2.Sobel` hoặc `cv2.filter2D`. Ảnh màu được chuyển sang xám bằng công thức trọng số BGR.

## 6. Sinh nhiễu

- Gaussian: lấy mẫu từ phân phối chuẩn `N(mean, sigma²)`, cộng vào pixel và cắt về [0,255].
- Salt & Pepper: chọn vị trí pixel bằng bộ sinh số giả ngẫu nhiên, thay pixel được chọn bằng 255 hoặc 0.
- Seed cố định giúp tái lập thực nghiệm. NumPy Generator chỉ đóng vai trò bộ sinh số ngẫu nhiên, không thực hiện thuật toán lọc.

## 7. Đánh giá chất lượng

- MSE: tự xây dựng mảng sai khác, bình phương từng phần tử, cộng tổng và chia cho số phần tử.
- PSNR: tính từ MSE theo `10 log10(255²/MSE)`; hai ảnh giống nhau có PSNR vô hạn.
- SSIM cục bộ: tự tính trung bình, phương sai và hiệp phương sai trên cửa sổ 7×7. Ảnh tích lũy (summed-area table) được xây dựng bằng phép cộng dồn theo hai trục; tổng cửa sổ được lấy từ bốn góc.
- Edge Precision/Recall/F1: tự tạo mặt nạ biên theo ngưỡng tỷ lệ gradient cực đại, sau đó đếm TP/FP/FN bằng vòng lặp.

## 8. Thứ tự pipeline

Ảnh tham chiếu → thêm nhiễu (nếu chọn) → Gamma → Contrast/Brightness → Mean/Median → tùy chọn hiển thị Sobel. Bản đồ biên được tính trên ảnh gốc, ảnh nhiễu, ảnh sau Mean và ảnh sau Median để khảo sát ảnh hưởng của lọc đến đường biên.

## 9. Phạm vi sử dụng thư viện

OpenCV được dùng để đọc/ghi ảnh và các thao tác I/O. NumPy được dùng để biểu diễn mảng, cấp phát bộ nhớ, đệm biên, chuyển kiểu và sinh nhiễu ngẫu nhiên. Các phép tính cốt lõi của Mean, Median, Sobel, MSE, PSNR và SSIM được viết tường minh trong mã nguồn. Đây là cách triển khai thuật toán độc lập trên nền tảng mảng số, không phải tự viết lại Python, NumPy hay bộ giải mã JPEG/PNG.


## Quy ước đánh giá chất lượng và đường biên

- MSE/PSNR: tính trên toàn bộ phần tử pixel của ảnh đầu vào (ảnh màu BGR gồm cả 3 kênh); PSNR dùng MAX=255 và được suy ra từ MSE.
- SSIM cục bộ: chuyển ảnh BGR sang ảnh xám theo trọng số luminance, tính mean/variance/covariance trên cửa sổ 7×7, sau đó lấy trung bình SSIM cục bộ.
- Edge Precision/Recall/F1: tạo gradient Sobel từ ảnh xám; ngưỡng nhị phân dùng chung cho ảnh tham chiếu và ảnh kết quả, bằng 15% cực đại gradient của ảnh tham chiếu. Cách dùng ngưỡng chung giúp đo cả việc lọc làm suy giảm biên độ gradient.
- Trước khi so sánh, ảnh tham chiếu sạch được áp dụng cùng Gamma/Contrast/Brightness như ảnh xử lý. Vì vậy metric đánh giá sai số sau lọc trong cùng miền cường độ.
