from datetime import datetime
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.image_io import write_image
from src.algorithms import (
    gamma_correction, adjust_contrast,
    add_gaussian_noise, add_salt_pepper_noise, mean_filter, median_filter,
    mse, psnr, ssim_local, edge_metrics
)


def run_experiment(clean_image, output_root="results/experiments", progress_callback=None):
    """Chạy 72 ca: 2 nhiễu x 2 mức x 2 bộ lọc x 3 kernel x 3 Gamma.

    Ảnh sạch đầu vào là ground truth. Mỗi Gamma được áp dụng giống nhau lên
    ground truth và ảnh nhiễu trước lọc để metric đo riêng sai số do nhiễu/lọc,
    không phạt thay đổi độ sáng có chủ đích của phép Gamma.
    """
    out_dir = Path(output_root) / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    noise_specs = [
        ("Gaussian", 12.0), ("Gaussian", 24.0),
        ("Salt & Pepper", 3.0), ("Salt & Pepper", 7.0),
    ]
    filter_specs = [("Mean", 3), ("Mean", 5), ("Mean", 7),
                    ("Median", 3), ("Median", 5), ("Median", 7)]
    gamma_specs = [0.8, 1.0, 1.2]
    total = len(noise_specs) * len(filter_specs) * len(gamma_specs)
    done = 0

    for noise_name, level in noise_specs:
        if noise_name == "Gaussian":
            noisy = add_gaussian_noise(clean_image, sigma=level, seed=2026)
        else:
            noisy = add_salt_pepper_noise(clean_image, amount=level / 100.0, seed=2026)

        for gamma in gamma_specs:
            # Ground truth và noisy image cùng Gamma; Contrast/Brightness giữ identity.
            reference_gamma = adjust_contrast(gamma_correction(clean_image, gamma), alpha=1.0, beta=0)
            noisy_gamma = adjust_contrast(gamma_correction(noisy, gamma), alpha=1.0, beta=0)
            noisy_quality = {
                "mse": mse(reference_gamma, noisy_gamma),
                "psnr": psnr(reference_gamma, noisy_gamma),
                "ssim": ssim_local(reference_gamma, noisy_gamma),
                "edge_f1": edge_metrics(reference_gamma, noisy_gamma)["f1"],
            }

            for filter_name, kernel in filter_specs:
                result = mean_filter(noisy_gamma, kernel) if filter_name == "Mean" else median_filter(noisy_gamma, kernel)
                quality = {
                    "mse": mse(reference_gamma, result),
                    "psnr": psnr(reference_gamma, result),
                    "ssim": ssim_local(reference_gamma, result),
                    "edge_f1": edge_metrics(reference_gamma, result)["f1"],
                }
                cases.append({
                    "noise": noise_name, "noise_level": level, "gamma": gamma,
                    "method": filter_name, "kernel": kernel,
                    "mse": quality["mse"], "psnr_db": quality["psnr"],
                    "ssim_local": quality["ssim"], "edge_f1": quality["edge_f1"],
                    "noisy_mse": noisy_quality["mse"],
                    "noisy_psnr_db": noisy_quality["psnr"],
                    "noisy_ssim_local": noisy_quality["ssim"],
                    "noisy_edge_f1": noisy_quality["edge_f1"],
                })
                done += 1
                if progress_callback:
                    progress_callback(done, total,
                        f"{noise_name} {level:g}% | Gamma {gamma:g} | {filter_name} {kernel}x{kernel}")
                if noise_name == "Salt & Pepper" and level == 7.0 and gamma == 1.0 and kernel == 3:
                    write_image(out_dir / "example_clean.png", reference_gamma)
                    write_image(out_dir / "example_noisy.png", noisy_gamma)
                    write_image(out_dir / f"example_{filter_name.lower()}.png", result)

    csv_path = out_dir / "experiment_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(cases[0].keys()))
        writer.writeheader()
        writer.writerows(cases)

    labels = [f"{r['noise'][0]}{r['noise_level']:g}-G{r['gamma']:.1f}-{r['method'][0]}{r['kernel']}" for r in cases]
    x = np.arange(len(cases))
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(x, [r["psnr_db"] for r in cases], marker="o", markersize=3, label="Sau lọc")
    ax.plot(x, [r["noisy_psnr_db"] for r in cases], linestyle="--", marker=".", markersize=2, label="Ảnh nhiễu")
    ax.set_title("PSNR trước và sau lọc theo loại nhiễu, Gamma và kernel")
    ax.set_ylabel("PSNR (dB)"); ax.set_xlabel("Nhiễu-mức / Gamma / Bộ lọc-kernel")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
    ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout()
    psnr_chart = out_dir / "psnr_comparison.png"
    fig.savefig(psnr_chart, dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(x, [r["edge_f1"] for r in cases], marker="o", markersize=3, label="F1 sau lọc")
    ax.plot(x, [r["noisy_edge_f1"] for r in cases], linestyle="--", marker=".", markersize=2, label="F1 ảnh nhiễu")
    ax.set_title("Ảnh hưởng của lọc và Gamma đến đường biên Sobel")
    ax.set_ylabel("Edge F1"); ax.set_xlabel("Nhiễu-mức / Gamma / Bộ lọc-kernel")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
    ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout()
    edge_chart = out_dir / "edge_f1_comparison.png"
    fig.savefig(edge_chart, dpi=150); plt.close(fig)

    return {"output_dir": str(out_dir.resolve()), "csv": str(csv_path.resolve()),
            "psnr_chart": str(psnr_chart.resolve()), "edge_chart": str(edge_chart.resolve()),
            "rows": cases}
