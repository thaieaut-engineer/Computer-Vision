import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.core.image_io import read_image, write_image
from src.core.pipeline import process_image
from src.core.experiment import run_experiment


class TestCoreServices(unittest.TestCase):
    def setUp(self):
        y, x = np.mgrid[0:48, 0:64]
        self.image = np.zeros((48, 64, 3), dtype=np.uint8)
        self.image[:, :, 0] = (x * 3).astype(np.uint8)
        self.image[:, :, 1] = (y * 4).astype(np.uint8)
        self.image[:, :, 2] = 120
        self.image[12:35, 18:45] = (230, 220, 210)

    def test_unicode_image_io(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ảnh kiểm thử.png"
            write_image(path, self.image)
            loaded = read_image(path)
            np.testing.assert_array_equal(loaded, self.image)

    def test_pipeline_returns_required_outputs(self):
        result = process_image(self.image, {
            "noise": "Salt & Pepper", "noise_level": 4,
            "gamma": 1.0, "contrast": 1.0, "brightness": 0,
            "filter": "Median", "kernel": 3, "sobel": "Không dùng",
        })
        self.assertEqual(result["enhanced"].shape, self.image.shape)
        self.assertTrue(result["quality"]["valid"])
        self.assertIn("mse", result["quality"])
        self.assertEqual(set(result["edge_panels"].keys()), {"Ảnh tham chiếu", "Ảnh nhiễu", "Sau Mean", "Sau Median"})

    def test_pipeline_without_noise_uses_original_as_reference(self):
        result = process_image(self.image, {
            "noise": "Không thêm nhiễu", "gamma": 1.0, "contrast": 1.0,
            "brightness": 0, "filter": "Median", "kernel": 3, "sobel": "Không dùng",
        })
        self.assertTrue(result["quality"]["valid"])
        self.assertEqual(result["quality"]["reference_type"], "Ảnh gốc")
        self.assertIn("mse", result["quality"])

    def test_separate_clean_reference_is_used(self):
        noisy = self.image.copy()
        noisy[::2, ::2] = 255 - noisy[::2, ::2]
        result = process_image(noisy, {
            "noise": "Không thêm nhiễu", "gamma": 1.0, "contrast": 1.0,
            "brightness": 0, "filter": "Median", "kernel": 3, "sobel": "Không dùng",
        }, reference_image=self.image)
        self.assertTrue(result["quality"]["valid"])
        self.assertEqual(result["quality"]["reference_type"], "Ảnh tham chiếu riêng")

    def test_reference_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            process_image(self.image, {"noise": "Không thêm nhiễu"}, reference_image=self.image[:20, :20])

    def test_experiment_creates_72_rows_and_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            result = run_experiment(self.image, output_root=temp)
            self.assertEqual(len(result["rows"]), 72)
            self.assertEqual({row["gamma"] for row in result["rows"]}, {0.8, 1.0, 1.2})
            self.assertTrue(Path(result["csv"]).exists())
            self.assertTrue(Path(result["psnr_chart"]).exists())
            self.assertTrue(Path(result["edge_chart"]).exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
