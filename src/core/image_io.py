from pathlib import Path
import cv2
import numpy as np


def read_image(path):
    """Đọc ảnh hỗ trợ cả đường dẫn Unicode trên Windows."""
    path = str(path)
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {path}")
    return image


def write_image(path, image):
    """Ghi ảnh hỗ trợ đường dẫn Unicode trên Windows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    if suffix not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        suffix = ".png"
        path = path.with_suffix(suffix)
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Không thể mã hóa ảnh định dạng {suffix}.")
    encoded.tofile(str(path))
    return path
