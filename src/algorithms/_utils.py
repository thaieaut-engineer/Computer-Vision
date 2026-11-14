import numpy as np


def validate_image(image):
    arr = np.asarray(image)
    if arr.ndim not in (2, 3) or arr.size == 0:
        raise ValueError("Ảnh phải là ma trận 2D hoặc 3D và không được rỗng.")
    if arr.dtype != np.uint8:
        raise TypeError("Ảnh đầu vào phải có kiểu uint8.")
    if arr.ndim == 3 and arr.shape[2] not in (1, 3):
        raise ValueError("Ảnh màu phải có 1 hoặc 3 kênh.")
    return arr


def validate_kernel_size(kernel_size):
    if not isinstance(kernel_size, (int, np.integer)) or kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("Kích thước kernel phải là số nguyên lẻ >= 3.")


def as_float_pair(reference, test):
    a, b = np.asarray(reference), np.asarray(test)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("Hai ảnh phải có cùng kích thước và không được rỗng.")
    return a.astype(np.float64), b.astype(np.float64)
