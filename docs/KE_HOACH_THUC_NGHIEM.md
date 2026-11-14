# Kế hoạch thực nghiệm

## 1. Mục tiêu

Đánh giá khả năng giảm nhiễu của Mean Filter và Median Filter, khảo sát ảnh hưởng kích thước kernel và phân tích mức bảo toàn đường biên Sobel.

## 2. Dữ liệu

Ảnh sạch tham chiếu phải có đủ chi tiết, vùng phẳng và đường biên. Có thể dùng `assets/sample_test.png` để kiểm tra ứng dụng; khi làm báo cáo nên bổ sung 5–10 ảnh đa dạng (chân dung, phong cảnh, kiến trúc, kết cấu). Ghi lại tên, kích thước và nguồn từng ảnh.

## 3. Thiết kế thí nghiệm

- Gaussian: sigma 12 và 24.
- Salt & Pepper: 3% và 7% pixel.
- Mean và Median.
- Kernel 3×3, 5×5, 7×7.
- Tổng: 72 cấu hình trên mỗi ảnh tham chiếu khi khảo sát thêm Gamma 0.8, 1.0, 1.2.
- Gamma khảo sát ở 0.8, 1.0, 1.2; Contrast=1 và Brightness=0. Ảnh tham chiếu cũng được biến đổi cùng Gamma trước khi tính chỉ số.

## 4. Chỉ số

Ghi MSE, PSNR, SSIM cục bộ và Edge F1. MSE thấp hơn, PSNR cao hơn, SSIM/Edge F1 gần 1 hơn thường là tín hiệu tốt hơn, nhưng cần xem cùng ảnh kết quả và đặc điểm nhiễu.

## 5. Bảng ghi kết quả

Dữ liệu chi tiết được xuất tự động trong `experiment_results.csv`. Trong báo cáo, có thể tổng hợp trung bình theo loại nhiễu, mức nhiễu, bộ lọc và kernel; trình bày thêm ảnh gốc, ảnh nhiễu, ảnh sau lọc và bản đồ biên Sobel.

## 6. Lưu ý khoa học

Ảnh tham chiếu phải là ảnh sạch. Nếu ảnh đầu vào vốn có nhiễu, không được gọi nó là ground truth. Với ảnh màu, SSIM cục bộ đang tính trên ảnh xám; cần nêu rõ trong báo cáo. Edge F1 là phép đo bổ sung dựa trên ngưỡng, không thay thế đánh giá trực quan hoặc các phân tích khác.


### Bổ sung: khảo sát Gamma

Mỗi ảnh nhiễu và ảnh gốc được biến đổi cùng Gamma (0.8, 1.0, 1.2) trước khi áp dụng Mean/Median. Ma trận thực nghiệm gồm 72 cấu hình: 2 loại nhiễu × 2 mức nhiễu × 3 Gamma × 2 bộ lọc × 3 kernel.


## 7. Chế độ dữ liệu thực nghiệm

Ảnh đầu vào được dùng làm ground truth. Chương trình tự sinh Gaussian hoặc Salt & Pepper theo seed cố định, sau đó lọc. MSE/PSNR/SSIM/Edge F1 được tính giữa ảnh sau lọc và ground truth đã biến đổi cùng Gamma.

## 8. Diễn giải chỉ số khi dùng Gamma

Với mỗi Gamma, chương trình áp dụng cùng phép biến đổi Gamma lên ảnh tham chiếu sạch và ảnh đầu vào trước khi so sánh. Cách này đo sai số còn lại sau lọc trong cùng miền cường độ, thay vì coi thay đổi độ sáng có chủ đích là lỗi. SSIM được tính trên ảnh xám theo cửa sổ cục bộ; MSE/PSNR tính trên các mẫu pixel của ảnh (bao gồm các kênh màu); Edge F1 so sánh bản đồ Sobel với ngưỡng tương đối theo cực đại từng bản đồ. Cần nêu rõ các quy ước này trong báo cáo.
