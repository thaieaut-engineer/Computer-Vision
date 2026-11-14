# Kết quả mẫu chạy trên ảnh đi kèm

Đây là kết quả demo tự động từ `assets/sample_test.png`, không phải số liệu thực nghiệm cuối cùng cho báo cáo. Hãy chạy lại trên bộ ảnh sạch do nhóm lựa chọn và lưu nguồn ảnh.

| Nhiễu | Mức | PSNR ảnh nhiễu | Cấu hình PSNR tốt nhất | PSNR sau lọc | SSIM sau lọc |
|---|---:|---:|---|---:|---:|
| Gaussian | 12.0 | 26.57 dB | Median 3×3 | 31.14 dB | 0.8614 |
| Gaussian | 24.0 | 20.63 dB | Median 5×5 | 28.09 dB | 0.8179 |
| Salt & Pepper | 3.0 | 20.53 dB | Median 3×3 | 33.59 dB | 0.9738 |
| Salt & Pepper | 7.0 | 16.81 dB | Median 3×3 | 32.39 dB | 0.9701 |

## Nhận xét minh họa

- Trên ảnh mẫu này, lọc giúp cải thiện PSNR so với ảnh nhiễu ở cả bốn điều kiện.
- Median 3×3 đạt PSNR cao nhất trong các cấu hình đã khảo sát trên ảnh mẫu này.
- Không được kết luận Median luôn tốt hơn Mean cho mọi ảnh; kết quả phụ thuộc loại nhiễu, nội dung ảnh và kích thước kernel.
- Kernel lớn không mặc nhiên tốt hơn: lọc mạnh có thể làm mất chi tiết và giảm độ tương đồng đường biên.
- Cần sử dụng thêm ảnh chân dung, phong cảnh, kiến trúc và ảnh có kết cấu để báo cáo có tính đại diện hơn.
