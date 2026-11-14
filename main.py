"""Chạy pipeline từ dòng lệnh, ngoài giao diện PyQt6."""
import argparse
from pathlib import Path
from src.core.image_io import read_image, write_image
from src.core.pipeline import process_image


def main():
    parser = argparse.ArgumentParser(description="ImageEnhancement – Đề tài xử lý ảnh số 4")
    parser.add_argument("--input", required=True, help="Đường dẫn ảnh cần xử lý")
    parser.add_argument("--reference", default=None, help="Ảnh tham chiếu sạch tương ứng (cùng kích thước/căn chỉnh)")
    parser.add_argument("--noise", choices=["none", "gaussian", "salt_pepper"], default="salt_pepper")
    parser.add_argument("--level", type=float, default=5.0, help="Gaussian: sigma; Salt&Pepper: phần trăm")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--brightness", type=int, default=0)
    parser.add_argument("--filter", choices=["none", "mean", "median"], default="median")
    parser.add_argument("--kernel", type=int, choices=[3, 5, 7], default=3)
    parser.add_argument("--sobel", choices=["none", "x", "y", "magnitude"], default="none")
    parser.add_argument("--output", default="results/cli_result.png")
    args = parser.parse_args()

    image = read_image(args.input)
    reference = read_image(args.reference) if args.reference else None
    noise = {"none": "Không thêm nhiễu", "gaussian": "Gaussian", "salt_pepper": "Salt & Pepper"}[args.noise]
    filter_name = {"none": "Không lọc", "mean": "Mean", "median": "Median"}[args.filter]
    sobel = {"none": "Không dùng", "x": "Sobel X", "y": "Sobel Y", "magnitude": "Sobel Magnitude"}[args.sobel]
    result = process_image(image, {
        "noise": noise, "noise_level": args.level, "gamma": args.gamma,
        "contrast": args.contrast, "brightness": args.brightness,
        "filter": filter_name, "kernel": args.kernel, "sobel": sobel,
    }, reference_image=reference)
    saved = write_image(args.output, result["display"])
    q = result["quality"]
    print(f"Đã lưu: {saved}")
    if q.get("valid"):
        print(f"MSE={q['mse']:.4f} | PSNR={q['psnr']:.4f} dB | SSIM local={q['ssim']:.6f}")
        print(f"Edge Precision={q['edge']['precision']:.6f} | Recall={q['edge']['recall']:.6f} | F1={q['edge']['f1']:.6f}")
        print(f"Ground truth: {q['reference_type']}")
    else:
        print("Không có ground truth hợp lệ; các chỉ số MSE/PSNR/SSIM/Edge F1 không được tính.")
        print(q["reason"])


if __name__ == "__main__":
    main()
