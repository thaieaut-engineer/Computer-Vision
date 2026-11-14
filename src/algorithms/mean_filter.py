import numpy as np
from ._utils import validate_image, validate_kernel_size


def _mean_gray(image, kernel_size):
    """Mean Filter tự cộng từng vị trí kernel, không dùng np.mean/cv2.blur."""
    height, width = image.shape
    radius = kernel_size // 2
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode="reflect")
    # Mỗi lần lặp cộng đúng một vị trí trong kernel vào ma trận tổng.
    total = np.zeros((height, width), dtype=np.uint64)
    for ky in range(kernel_size):
        for kx in range(kernel_size):
            total += padded[ky:ky + height, kx:kx + width].astype(np.uint64)
    divisor = kernel_size * kernel_size
    # Làm tròn half-up bằng số nguyên, không dùng hàm lọc ảnh có sẵn.
    return ((total + divisor // 2) // divisor).clip(0, 255).astype(np.uint8)


def mean_filter(image, kernel_size=3):
    """Mean Filter tự cài đặt; xử lý từng kênh màu độc lập."""
    image = validate_image(image)
    validate_kernel_size(kernel_size)
    if image.ndim == 2:
        return _mean_gray(image, kernel_size)
    if image.shape[2] == 1:
        return _mean_gray(image[:, :, 0], kernel_size)[:, :, None]
    return np.stack([_mean_gray(image[:, :, c], kernel_size) for c in range(3)], axis=2)
