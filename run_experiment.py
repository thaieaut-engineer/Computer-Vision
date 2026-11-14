import argparse
from src.core.image_io import read_image
from src.core.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Chạy bộ thực nghiệm 72 ca của ImageEnhancement")
    parser.add_argument("--input", required=True, help="Ảnh sạch tham chiếu (ground truth; không dùng ảnh đã nhiễu)")
    parser.add_argument("--output", default="results/experiments")
    args = parser.parse_args()
    image = read_image(args.input)
    result = run_experiment(image, args.output,
                            progress_callback=lambda done, total, label: print(f"[{done}/{total}] {label}"))
    print("Hoàn tất. Kết quả:", result["output_dir"])
    print("CSV:", result["csv"])
    print("Biểu đồ PSNR:", result["psnr_chart"])
    print("Biểu đồ Sobel:", result["edge_chart"])


if __name__ == "__main__":
    main()
