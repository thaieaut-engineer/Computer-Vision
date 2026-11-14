import numpy as np
from ._utils import validate_image


def add_gaussian_noise(image, mean=0.0, sigma=20.0, seed=None):
    """Thêm nhiễu Gaussian cộng; sigma được tính theo thang 0..255."""
    image = validate_image(image)
    if sigma < 0:
        raise ValueError("Sigma không được âm.")
    rng = np.random.default_rng(seed)
    noise = rng.normal(float(mean), float(sigma), size=image.shape)
    return np.clip(np.rint(image.astype(np.float64) + noise), 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image, amount=0.05, salt_vs_pepper=0.5, seed=None):
    """Thay ngẫu nhiên amount tỷ lệ pixel bằng điểm trắng/đen."""
    image = validate_image(image)
    if not 0 <= amount <= 1 or not 0 <= salt_vs_pepper <= 1:
        raise ValueError("amount và salt_vs_pepper phải thuộc [0, 1].")
    rng = np.random.default_rng(seed)
    result = image.copy()
    h, w = image.shape[:2]
    count = int(round(amount * h * w))
    if count == 0:
        return result
    indices = rng.choice(h * w, size=min(count, h * w), replace=False)
    salt_count = int(round(len(indices) * salt_vs_pepper))
    ys, xs = np.divmod(indices, w)
    result[ys[:salt_count], xs[:salt_count]] = 255
    result[ys[salt_count:], xs[salt_count:]] = 0
    return result
