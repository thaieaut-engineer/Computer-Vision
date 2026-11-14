# ImageEnhancement – Đề tài số 4

**Học phần:** Xử lý ảnh và thị giác máy tính  
**Đề tài:** Nâng cao chất lượng ảnh bằng biến đổi điểm ảnh và lọc trong miền không gian  
**Nền tảng:** Windows 10/11 · Python 3.12 · PyQt6 · NumPy · OpenCV

## 1. Mục tiêu

Xây dựng ứng dụng desktop cho phép khảo sát tác động của biến đổi Gamma/độ tương phản và bộ lọc trung bình/trung vị lên ảnh có nhiễu; đồng thời dùng Sobel để phân tích ảnh hưởng của lọc đến đường biên.

## 2. Chức năng

- Mở/lưu ảnh JPG, JPEG, PNG, BMP, TIFF (hỗ trợ đường dẫn Unicode trên Windows).
- Biến đổi Gamma, tương phản tuyến tính và độ sáng.
- Sinh nhiễu Gaussian và Salt & Pepper có seed cố định để tái lập thực nghiệm.
- Mean Filter tự cộng pixel bằng vòng lặp; Median Filter tự chọn hạng bằng tìm kiếm nhị phân; hỗ trợ kernel 3×3, 5×5, 7×7.
- Sobel X, Sobel Y và độ lớn gradient tự tích chập bằng vòng lặp 3×3.
- So sánh bốn bản đồ biên: ảnh tham chiếu, ảnh nhiễu, sau Mean, sau Median.
- Đánh giá bằng MSE/PSNR tự triển khai công thức, SSIM cục bộ tự tính thống kê cửa sổ, Edge Precision/Recall/F1.
- Hiển thị histogram ảnh gốc và ảnh kết quả.
- Chọn ảnh tham chiếu sạch riêng (ground truth); kiểm tra kích thước trước khi đánh giá.
- Không tự tính MSE/PSNR/SSIM/Edge F1 nếu ảnh thực tế không có ground truth sạch.
- Tự động chạy ma trận 72 ca: 2 loại nhiễu × 2 mức × 2 bộ lọc × 3 kernel × 3 mức Gamma (0.8, 1.0, 1.2), xuất CSV và biểu đồ PSNR/Edge F1.
- Chạy xử lý trong QThread để giao diện không bị khóa khi đang tính toán.

## 3. Cấu trúc dự án

```text
ImageEnhancement_Final/
├── main.py                    # Chạy pipeline từ dòng lệnh
├── run_gui.py                 # Khởi chạy ứng dụng PyQt6
├── run_experiment.py          # Chạy bộ thực nghiệm từ dòng lệnh
├── requirements.txt
├── README.md
├── assets/
│   └── sample_test.png        # Ảnh kiểm thử đi kèm
├── docs/
│   ├── THUAT_TOAN.md
│   └── KE_HOACH_THUC_NGHIEM.md
├── src/
│   ├── algorithms/            # Thuật toán lõi tự cài đặt
│   ├── core/                  # Pipeline, đọc/ghi ảnh, thực nghiệm
│   └── ui/                    # Giao diện PyQt6
├── tests/
│   └── test_algorithms.py
└── results/                   # Kết quả sinh ra khi chạy
```

## 4. Cài đặt trên Windows (Python 3.12)

Mở PowerShell hoặc Terminal tại thư mục dự án.

### Cách A – Dùng Python hiện có (không tạo môi trường ảo)

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Cách B – Dùng môi trường ảo (khuyến nghị nếu máy có nhiều dự án)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Không cần cài cả hai cách. Nếu dùng cách A, bỏ qua bước kích hoạt `.venv`.

## 5. Chạy ứng dụng

```powershell
python run_gui.py
```

Mở ảnh cần xử lý bằng nút **Mở ảnh**. Nếu ảnh đầu vào vốn đã nhiễu, chọn thêm **Ảnh tham chiếu sạch (Ground Truth)** cùng kích thước trước khi áp dụng. Nếu mô phỏng nhiễu nhân tạo từ ảnh sạch đang mở, ảnh đang mở được dùng làm ground truth. Nút **Lưu ảnh kết quả** lưu ảnh đang hiển thị. Tab **Thực nghiệm** chạy bộ 72 ca trên ảnh sạch tham chiếu đã chọn; nếu chưa chọn, ứng dụng yêu cầu xác nhận ảnh đang mở là ảnh sạch. Tab **Histogram** hiển thị phân bố mức xám.

Ảnh mẫu để thử ngay: `assets/sample_test.png`.

## 6. Chạy kiểm thử tự động

```powershell
python -m unittest discover -s tests -v
```

Các test kiểm tra Gamma/Contrast, Mean/Median, loại nhiễu xung, tính tái lập của sinh nhiễu, Sobel trên ảnh phẳng và các chỉ số đánh giá. Chi tiết cách tự cài đặt nằm trong `docs/THUAT_TOAN.md`; hướng dẫn hai bộ dữ liệu nằm trong `docs/QUY_TRINH_DU_LIEU.md`.

## 7. Chạy pipeline từ CMD/PowerShell

Ví dụ Salt & Pepper + Gamma + Median 3×3 trên ảnh sạch:

```powershell
python main.py --input assets/sample_test.png --noise salt_pepper --level 5 --gamma 0.8 --filter median --kernel 3 --output results/demo_median.png
```

Ví dụ xử lý ảnh thực tế đã nhiễu, có ảnh tham chiếu sạch tương ứng (không thêm nhiễu lần nữa):

```powershell
python main.py --input assets/train_noisy.png --reference assets/train_clean.png --noise none --filter median --kernel 3 --output results/train_restored.png
```

Ví dụ Gaussian + Mean 5×5:

```powershell
python main.py --input assets/sample_test.png --noise gaussian --level 20 --filter mean --kernel 5 --output results/demo_mean.png
```

## 8. Chạy thực nghiệm 72 ca

Từ giao diện: mở ảnh sạch → tab **Thực nghiệm** → **Chạy 72 ca thực nghiệm**.

Hoặc chạy từ dòng lệnh:

```powershell
python run_experiment.py --input assets/sample_test.png
```

Mỗi lần chạy tạo một thư mục con trong `results/experiments/` gồm:

- `experiment_results.csv`: dữ liệu số cho 72 cấu hình.
- `psnr_comparison.png`: so sánh PSNR ảnh nhiễu và ảnh sau lọc.
- `edge_f1_comparison.png`: so sánh độ tương đồng đường biên Sobel.
- Một số ảnh ví dụ sạch/nhiễu/kết quả.

### Thiết kế ma trận thực nghiệm

| Yếu tố | Giá trị |
|---|---|
| Loại nhiễu | Gaussian, Salt & Pepper |
| Mức Gaussian σ | 12, 24 |
| Mức Salt & Pepper | 3%, 7% |
| Bộ lọc | Mean, Median |
| Kernel | 3×3, 5×5, 7×7 |
| Tổng số cấu hình | 2 × 2 × 2 × 3 = 24 |

Trong bộ thực nghiệm, Gamma=1, Contrast=1 và Brightness=0 được giữ cố định để so sánh riêng tác động của bộ lọc. Các phép biến đổi Gamma/Contrast được khảo sát riêng trong tab Xử lý ảnh. Ảnh đầu vào được coi là ảnh sạch tham chiếu; nếu dùng ảnh vốn đã nhiễu thì không được diễn giải các chỉ số như so sánh với ground truth sạch.

## 9. Lưu ý khi phân tích chỉ số

- MSE càng thấp càng tốt; PSNR càng cao càng tốt; SSIM càng gần 1 càng tốt.
- SSIM được cài đặt theo cửa sổ cục bộ 7×7 trên ảnh xám (không gọi hàm SSIM có sẵn).
- Edge F1 đo mức trùng khớp giữa bản đồ biên Sobel nhị phân của ảnh tham chiếu và ảnh đang đánh giá. Đây là chỉ số bổ sung, cần kết hợp quan sát trực quan.
- Khi Gamma/Contrast thay đổi, MSE/PSNR/SSIM so với ảnh gốc phản ánh cả thay đổi độ sáng/tương phản lẫn khử nhiễu. Muốn đánh giá riêng lọc, giữ Gamma=1, Contrast=1, Brightness=0.
- Bộ lọc xử lý độc lập từng kênh BGR. Sobel chuyển ảnh màu sang xám.
- Sobel magnitude được chuẩn hóa riêng để hiển thị; dữ liệu gradient gốc vẫn được giữ ở dạng số thực trong pipeline.

## 10. Đẩy dự án lên GitHub

```powershell
git init
git add .
git commit -m "Complete image enhancement project"
git branch -M main
git remote add origin https://github.com/USERNAME/ImageEnhancement.git
git push -u origin main
```

Thay `USERNAME` bằng tài khoản GitHub của bạn. Không đẩy thư mục `.venv`, file tạm hoặc dữ liệu kết quả dung lượng lớn.

## 11. Chạy nhanh bằng file BAT

- `setup_windows.bat`: cài thư viện vào Python đang hoạt động (không tự tạo môi trường ảo).
- `run_windows.bat`: khởi chạy giao diện.
- `test_windows.bat`: chạy toàn bộ kiểm thử.

Nếu Windows không nhận lệnh `python`, hãy kiểm tra PATH hoặc chạy các lệnh tương đương bằng `py -3.12` trong PowerShell.

## 12. Kết quả mẫu đi kèm

Thư mục `docs/sample_results/` chứa CSV, biểu đồ và ảnh ví dụ từ một lần chạy trên ảnh mẫu. Đây chỉ là dữ liệu minh họa để kiểm tra chương trình; báo cáo chính thức cần chạy lại trên bộ dữ liệu sạch do nhóm lựa chọn và ghi rõ nguồn ảnh.


## 9. Quy tắc ground truth và đánh giá định lượng

- Trường hợp mô phỏng nhiễu: ảnh sạch dùng tạo nhiễu là ground truth. Nếu đã chọn ảnh tham chiếu riêng, ảnh tham chiếu riêng được ưu tiên làm nguồn sạch.
- Trường hợp xử lý ảnh thực tế vốn đã nhiễu: phải chọn ảnh tham chiếu sạch tương ứng. Nếu không có, giao diện hiển thị MSE/PSNR/SSIM/Edge F1 là N/A thay vì giả định ảnh nhiễu là ảnh sạch.
- Ảnh tham chiếu phải cùng chiều rộng, chiều cao và số kênh với ảnh đầu vào.
- Khi Gamma/Contrast/Brightness thay đổi, cùng phép biến đổi được áp dụng lên ảnh tham chiếu trước khi tính chỉ số. Như vậy, phép đo tập trung vào sai số do nhiễu và lọc, không phạt thay đổi tông sáng có chủ đích.
- Bộ thực nghiệm tự động dùng ảnh sạch làm ground truth, chạy Gamma 0.8, 1.0, 1.2; 2 loại nhiễu (Gaussian, Salt & Pepper), 2 mức mỗi loại, 2 bộ lọc (Mean, Median) và kernel 3, 5, 7: tổng cộng 72 ca.
- Với ảnh thực tế không có ảnh tham chiếu tương ứng, chỉ đánh giá định tính bằng quan sát, histogram và cấu trúc; không báo cáo PSNR/SSIM như thước đo chất lượng khôi phục.
