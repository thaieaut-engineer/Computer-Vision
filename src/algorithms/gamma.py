import numpy as np
from ._utils import validate_image


def gamma_correction(image, gamma):
    """Biến đổi gamma: s = 255 * (r / 255) ** gamma."""
    image = validate_image(image)
    if gamma <= 0:
        raise ValueError("Gamma phải lớn hơn 0.")
    normalized = image.astype(np.float64) / 255.0
    result = 255.0 * np.power(normalized, float(gamma))
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)
