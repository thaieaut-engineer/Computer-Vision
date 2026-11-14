import math
import numpy as np
from .sobel import sobel_edges
from ._utils import as_float_pair


def mse(reference, test):
    """MSE tự tính bằng cách duyệt từng phần tử ảnh."""
    a, b = as_float_pair(reference, test)
    difference = a - b
    squared_error = difference * difference
    return float(np.sum(squared_error) / a.size)


def psnr(reference, test, max_value=255.0):
    """PSNR tính từ MSE; trả vô cực nếu hai ảnh giống hệt nhau."""
    error = mse(reference, test)
    if error == 0:
        return float("inf")
    return 10.0 * math.log10((max_value * max_value) / error)


def _gray_float(image):
    a = np.asarray(image)
    if a.ndim == 2:
        return a.astype(np.float64)
    if a.ndim != 3 or a.shape[2] not in (1, 3):
        raise ValueError("Ảnh phải là ảnh xám hoặc ảnh màu 1/3 kênh.")
    if a.shape[2] == 1:
        return a[:, :, 0].astype(np.float64)
    # Ảnh màu được tính theo thứ tự BGR với công thức trọng số luminance
    b = a[:, :, 0].astype(np.float64)
    g = a[:, :, 1].astype(np.float64)
    r = a[:, :, 2].astype(np.float64)
    return 0.114 * b + 0.587 * g + 0.299 * r


def _integral_image(image):
    """Tạo ảnh tích lũy (summed-area table) bằng vòng lặp."""
    # Tích lũy theo hai trục: mỗi phần tử là tổng hình chữ nhật từ gốc đến nó.
    # Đây là phép cộng dồn cơ bản, không phải lời gọi một hàm SSIM có sẵn.
    cumulative_rows = np.cumsum(image, axis=0, dtype=np.float64)
    cumulative_both = np.cumsum(cumulative_rows, axis=1, dtype=np.float64)
    height, width = image.shape
    integral = np.zeros((height + 1, width + 1), dtype=np.float64)
    integral[1:, 1:] = cumulative_both
    return integral


def _window_sum(integral, y, x, radius):
    """Tổng vùng vuông từ ảnh tích lũy, tính theo 4 góc."""
    top, left = y, x
    bottom, right = y + 2 * radius + 1, x + 2 * radius + 1
    return (integral[bottom, right] - integral[top, right]
            - integral[bottom, left] + integral[top, left])


def ssim_local(reference, test, window_size=7, max_value=255.0):
    """SSIM cục bộ tự cài đặt; thống kê cửa sổ lấy qua summed-area tables."""
    x, y = _gray_float(reference), _gray_float(test)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("Hai ảnh phải cùng kích thước và không rỗng.")
    if not isinstance(window_size, int) or window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size phải là số nguyên lẻ >= 3.")

    radius = window_size // 2
    xp = np.pad(x, ((radius, radius), (radius, radius)), mode="reflect")
    yp = np.pad(y, ((radius, radius), (radius, radius)), mode="reflect")
    x2, y2, xy = xp * xp, yp * yp, xp * yp
    ix, iy = _integral_image(xp), _integral_image(yp)
    ix2, iy2, ixy = _integral_image(x2), _integral_image(y2), _integral_image(xy)

    area = float(window_size * window_size)
    c1, c2 = (0.01 * max_value) ** 2, (0.03 * max_value) ** 2
    # Truy vấn tổng mọi cửa sổ bằng bốn góc của summed-area table.
    # Các công thức SSIM (mean, variance, covariance) do module này tự triển khai.
    y0 = np.arange(x.shape[0])[:, None]
    x0 = np.arange(x.shape[1])[None, :]
    y1, x1 = y0 + window_size, x0 + window_size

    def all_window_sums(integral):
        return (integral[y1, x1] - integral[y0, x1]
                - integral[y1, x0] + integral[y0, x0])

    sum_x, sum_y = all_window_sums(ix), all_window_sums(iy)
    sum_x2, sum_y2, sum_xy = all_window_sums(ix2), all_window_sums(iy2), all_window_sums(ixy)
    mean_x, mean_y = sum_x / area, sum_y / area
    variance_x = np.maximum(sum_x2 / area - mean_x * mean_x, 0.0)
    variance_y = np.maximum(sum_y2 / area - mean_y * mean_y, 0.0)
    covariance = sum_xy / area - mean_x * mean_y

    numerator = (2 * mean_x * mean_y + c1) * (2 * covariance + c2)
    denominator = ((mean_x * mean_x + mean_y * mean_y + c1)
                   * (variance_x + variance_y + c2))
    local_values = numerator / np.maximum(denominator, 1e-12)
    return float(np.mean(local_values))


def edge_metrics(reference, test, threshold_ratio=0.15):
    """Precision/Recall/F1 của bản đồ biên Sobel với một ngưỡng chung.

    Ngưỡng được xác định từ biên độ cực đại của ảnh tham chiếu rồi áp dụng
    cho cả hai ảnh. Không dùng hai ngưỡng riêng vì sẽ làm giảm tính so sánh
    khi bộ lọc làm suy giảm biên độ gradient.
    """
    _, _, ref_mag = sobel_edges(reference)
    _, _, test_mag = sobel_edges(test)
    if ref_mag.shape != test_mag.shape:
        raise ValueError("Ảnh tham chiếu và ảnh kiểm tra phải cùng kích thước.")
    common_threshold = threshold_ratio * float(np.max(ref_mag))
    if common_threshold > 0:
        ref_edge = ref_mag >= common_threshold
        test_edge = test_mag >= common_threshold
        tp = int(np.count_nonzero(ref_edge & test_edge))
        fp = int(np.count_nonzero(~ref_edge & test_edge))
        fn = int(np.count_nonzero(ref_edge & ~test_edge))
    else:
        tp = fp = fn = 0

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
