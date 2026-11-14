import numpy as np
from ._utils import validate_image, validate_kernel_size


def _mean_gray(image, kernel_size):
    """Mean Filter tự cài đặt bằng tích chập 1D tách biệt (Separable Filter).

    Toán học: Kernel 2D box filter K = (1/k²) · 1_(k×k) được phân rã thành hai bộ lọc 1D:
    dọc (k×1) và ngang (1×k). Độ phức tạp giảm từ O(k²) xuống O(2k).
    Phép cộng dồn bảo toàn tính chính xác số học nguyên tuyệt đối.
    Không dùng scipy.ndimage, cv2.blur hay np.mean.
    """
    height, width = image.shape
    radius = kernel_size // 2
    # Đệm biên bằng phản xạ (reflect padding) giống hệt phiên bản 2D gốc
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode="reflect").astype(np.uint64)

    # Bước 1: Lọc 1D theo phương ngang (cộng k cột lân cận)
    horiz = np.zeros((padded.shape[0], width), dtype=np.uint64)
    for kx in range(kernel_size):
        horiz += padded[:, kx:kx + width]

    # Bước 2: Lọc 1D theo phương dọc (cộng k hàng lân cận)
    total = np.zeros((height, width), dtype=np.uint64)
    for ky in range(kernel_size):
        total += horiz[ky:ky + height, :]

    divisor = kernel_size * kernel_size
    # Làm tròn half-up số nguyên chuẩn
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

