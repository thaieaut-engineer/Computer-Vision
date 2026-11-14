import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from ._utils import validate_image, validate_kernel_size


def insertion_sort(values):
    """Thuật toán Insertion Sort tham chiếu, dùng để minh họa/sanity check."""
    data = list(values)
    for i in range(1, len(data)):
        key = data[i]
        j = i - 1
        while j >= 0 and data[j] > key:
            data[j + 1] = data[j]
            j -= 1
        data[j + 1] = key
    return data


def _median_gray(image, kernel_size, chunk_rows=24):
    """Median bằng thuật toán chọn hạng (rank selection) + tìm kiếm nhị phân.

    Với mỗi cửa sổ, tìm giá trị nhỏ nhất v sao cho có hơn k²//2 phần tử <= v.
    Không gọi np.median, np.partition, np.sort hoặc cv2.medianBlur.
    """
    height, width = image.shape
    radius = kernel_size // 2
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode="reflect")
    windows = sliding_window_view(padded, (kernel_size, kernel_size))
    result = np.empty((height, width), dtype=np.uint8)
    target_rank = (kernel_size * kernel_size) // 2

    # Tìm kiếm nhị phân trên miền giá trị pixel uint8 [0, 255].
    for y0 in range(0, height, chunk_rows):
        y1 = min(y0 + chunk_rows, height)
        block = windows[y0:y1]
        low = np.zeros((y1 - y0, width), dtype=np.int16)
        high = np.full((y1 - y0, width), 255, dtype=np.int16)
        while np.any(low < high):
            active = low < high
            middle = (low + high) // 2
            counts = np.count_nonzero(block <= middle[:, :, None, None], axis=(-1, -2))
            move_high = active & (counts > target_rank)
            move_low = active & ~move_high
            high[move_high] = middle[move_high]
            low[move_low] = middle[move_low] + 1
        result[y0:y1] = low.astype(np.uint8)
    return result


def median_filter(image, kernel_size=3):
    """Median Filter tự cài đặt theo rank selection, xử lý từng kênh màu."""
    image = validate_image(image)
    validate_kernel_size(kernel_size)
    if image.ndim == 2:
        return _median_gray(image, kernel_size)
    if image.shape[2] == 1:
        return _median_gray(image[:, :, 0], kernel_size)[:, :, None]
    channels = [_median_gray(image[:, :, c], kernel_size) for c in range(3)]
    return np.stack(channels, axis=2)
