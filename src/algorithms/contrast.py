import numpy as np
from ._utils import validate_image


def adjust_contrast(image, alpha=1.0, beta=0):
    """Biến đổi tuyến tính: g(x,y) = alpha*f(x,y) + beta."""
    image = validate_image(image)
    if alpha < 0:
        raise ValueError("Hệ số tương phản alpha không được âm.")
    result = float(alpha) * image.astype(np.float64) + float(beta)
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)
