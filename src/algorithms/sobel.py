import numpy as np
from ._utils import validate_image

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def to_grayscale(image):
    """Chuyển ảnh BGR sang xám theo công thức trọng số luminance."""
    image = validate_image(image)
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    b = image[:, :, 0].astype(np.float64)
    g = image[:, :, 1].astype(np.float64)
    r = image[:, :, 2].astype(np.float64)
    return np.clip(np.rint(0.114 * b + 0.587 * g + 0.299 * r), 0, 255).astype(np.uint8)


def convolve3(image, kernel):
    """Tích chập 3x3 tự cài đặt bằng tổng 9 vùng dịch; biên dùng reflect."""
    source = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("convolve3 chỉ nhận ảnh xám 2D.")
    if kernel.shape != (3, 3):
        raise ValueError("Kernel tích chập phải có kích thước 3x3.")

    height, width = source.shape
    padded = np.pad(source, ((1, 1), (1, 1)), mode="reflect")
    result = np.zeros((height, width), dtype=np.float64)
    # Tự thực hiện từng phép nhân hệ số kernel và cộng dồn.
    for ky in range(3):
        for kx in range(3):
            result += kernel[ky, kx] * padded[ky:ky + height, kx:kx + width]
    return result


def sobel_edges(image):
    """Trả về gradient Gx, Gy và độ lớn gradient chưa chuẩn hóa."""
    gray = to_grayscale(image)
    gx = convolve3(gray, SOBEL_X)
    gy = convolve3(gray, SOBEL_Y)
    magnitude = np.sqrt(gx * gx + gy * gy)
    return gx, gy, magnitude


def sobel_display(gradient, absolute=True):
    """Chuẩn hóa gradient về uint8 chỉ để hiển thị."""
    values = np.abs(gradient) if absolute else np.asarray(gradient)
    maximum = float(np.max(values))
    if maximum <= 0:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip(np.rint(values * (255.0 / maximum)), 0, 255).astype(np.uint8)
