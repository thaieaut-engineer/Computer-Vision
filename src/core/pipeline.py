from src.algorithms import (
    gamma_correction, adjust_contrast, mean_filter, median_filter,
    add_gaussian_noise, add_salt_pepper_noise, sobel_edges, sobel_display,
    mse, psnr, ssim_local, edge_metrics
)


def apply_noise(image, noise_type="Không thêm nhiễu", noise_level=5.0, seed=42):
    if noise_type == "Gaussian":
        return add_gaussian_noise(image, sigma=float(noise_level), seed=seed)
    if noise_type == "Salt & Pepper":
        return add_salt_pepper_noise(image, amount=float(noise_level) / 100.0, seed=seed)
    return image.copy()


def _transform_reference(reference, settings):
    """Áp dụng cùng Gamma/Contrast/Brightness lên ground truth."""
    result = gamma_correction(reference, settings.get("gamma", 1.0))
    return adjust_contrast(
        result,
        alpha=settings.get("contrast", 1.0),
        beta=settings.get("brightness", 0),
    )


def process_image(original, settings, reference_image=None):
    """Xử lý ảnh và tính các chỉ số đánh giá chất lượng (MSE, PSNR, SSIM, Edge F1)."""
    noise_name = settings.get("noise", "Không thêm nhiễu")
    synthetic_noise = noise_name != "Không thêm nhiễu"

    if reference_image is not None:
        if reference_image.shape != original.shape:
            raise ValueError(
                "Ảnh đầu vào và ảnh tham chiếu phải cùng kích thước và số kênh. "
                f"Đầu vào: {original.shape}, tham chiếu: {reference_image.shape}."
            )

    clean_reference = reference_image.copy() if reference_image is not None else original.copy()

    if synthetic_noise:
        noisy = apply_noise(
            original, noise_name,
            settings.get("noise_level", 5.0), settings.get("seed", 42)
        )
    else:
        noisy = original.copy()

    transformed = _transform_reference(noisy, settings)
    filter_name = settings.get("filter", "Không lọc")
    kernel = int(settings.get("kernel", 3))
    if filter_name == "Mean":
        enhanced = mean_filter(transformed, kernel)
    elif filter_name == "Median":
        enhanced = median_filter(transformed, kernel)
    else:
        enhanced = transformed.copy()

    gx, gy, magnitude = sobel_edges(enhanced)
    sobel_mode = settings.get("sobel", "Không dùng")
    if sobel_mode == "Sobel X":
        display = sobel_display(gx)
    elif sobel_mode == "Sobel Y":
        display = sobel_display(gy)
    elif sobel_mode == "Sobel Magnitude":
        display = sobel_display(magnitude, absolute=False)
    else:
        display = enhanced.copy()

    # So sánh ảnh kết quả với ảnh tham chiếu/gốc đã trải qua cùng biến đổi điểm ảnh.
    transformed_reference = _transform_reference(clean_reference, settings)
    quality = {
        "valid": True,
        "reference_type": "Ảnh tham chiếu riêng" if reference_image is not None else "Ảnh gốc",
        "mse": mse(transformed_reference, enhanced),
        "psnr": psnr(transformed_reference, enhanced),
        "ssim": ssim_local(transformed_reference, enhanced),
        "edge": edge_metrics(transformed_reference, enhanced),
        "reason": "",
    }

    # Sobel panels dùng cùng biến đổi tông sáng; chỉ khác trạng thái lọc.
    edge_base = transformed
    mean_for_edges = mean_filter(edge_base, kernel)
    median_for_edges = median_filter(edge_base, kernel)
    edge_reference_source = transformed_reference
    _, _, edge_reference = sobel_edges(edge_reference_source)
    _, _, edge_noisy = sobel_edges(edge_base)
    _, _, edge_mean = sobel_edges(mean_for_edges)
    _, _, edge_median = sobel_edges(median_for_edges)
    edge_panels = {
        "Ảnh tham chiếu": sobel_display(edge_reference, absolute=False),
        "Ảnh nhiễu": sobel_display(edge_noisy, absolute=False),
        "Sau Mean": sobel_display(edge_mean, absolute=False),
        "Sau Median": sobel_display(edge_median, absolute=False),
    }
    return {
        "noisy": noisy, "enhanced": enhanced, "display": display,
        "gx": gx, "gy": gy, "magnitude": magnitude,
        "quality": quality, "edge_panels": edge_panels,
    }
