import unittest
import numpy as np
from src.algorithms import (
    gamma_correction, adjust_contrast, mean_filter, median_filter,
    add_gaussian_noise, add_salt_pepper_noise, sobel_edges,
    mse, psnr, ssim_local, edge_metrics,
)


class TestImageAlgorithms(unittest.TestCase):
    def setUp(self):
        self.constant = np.full((15, 15), 100, dtype=np.uint8)

    def test_gamma_identity(self):
        np.testing.assert_array_equal(gamma_correction(self.constant, 1.0), self.constant)

    def test_gamma_below_one_brightens_midtones(self):
        self.assertGreater(int(gamma_correction(self.constant, 0.5)[0, 0]), 100)

    def test_gamma_rejects_zero(self):
        with self.assertRaises(ValueError):
            gamma_correction(self.constant, 0)

    def test_contrast_identity(self):
        np.testing.assert_array_equal(adjust_contrast(self.constant, 1.0, 0), self.constant)

    def test_contrast_clips(self):
        image = np.array([[0, 100, 255]], dtype=np.uint8)
        result = adjust_contrast(image, 2.0, 10)
        self.assertEqual(result.tolist(), [[10, 210, 255]])

    def test_mean_constant(self):
        np.testing.assert_array_equal(mean_filter(self.constant, 3), self.constant)

    def test_mean_reduces_impulse(self):
        image = self.constant.copy()
        image[7, 7] = 255
        self.assertLess(int(mean_filter(image, 3)[7, 7]), 255)

    def test_median_removes_impulse(self):
        image = self.constant.copy()
        image[7, 7] = 255
        self.assertEqual(int(median_filter(image, 3)[7, 7]), 100)

    def test_insertion_sort(self):
        from src.algorithms.median_filter import insertion_sort
        self.assertEqual(insertion_sort([9, 1, 5, 1, 3]), [1, 1, 3, 5, 9])

    def test_sobel_detects_vertical_step(self):
        image = np.zeros((7, 7), dtype=np.uint8)
        image[:, 4:] = 255
        gx, gy, magnitude = sobel_edges(image)
        self.assertGreater(float(np.max(np.abs(gx))), 0.0)
        self.assertGreater(float(magnitude.max()), 0.0)

    def test_median_color_shape(self):
        image = np.full((12, 13, 3), 80, dtype=np.uint8)
        self.assertEqual(median_filter(image, 5).shape, image.shape)

    def test_invalid_kernel(self):
        with self.assertRaises(ValueError):
            mean_filter(self.constant, 4)

    def test_gaussian_noise_reproducible(self):
        a = add_gaussian_noise(self.constant, sigma=10, seed=17)
        b = add_gaussian_noise(self.constant, sigma=10, seed=17)
        np.testing.assert_array_equal(a, b)

    def test_salt_pepper_reproducible_and_range(self):
        a = add_salt_pepper_noise(self.constant, amount=0.2, seed=9)
        b = add_salt_pepper_noise(self.constant, amount=0.2, seed=9)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(set(np.unique(a)).issubset({0, 100, 255}))

    def test_sobel_constant_has_no_edges(self):
        gx, gy, mag = sobel_edges(self.constant)
        self.assertEqual(float(np.max(np.abs(gx))), 0.0)
        self.assertEqual(float(np.max(np.abs(gy))), 0.0)
        self.assertEqual(float(mag.max()), 0.0)

    def test_metrics_identical(self):
        self.assertEqual(mse(self.constant, self.constant), 0.0)
        self.assertEqual(psnr(self.constant, self.constant), float("inf"))
        self.assertAlmostEqual(ssim_local(self.constant, self.constant), 1.0, places=6)
        self.assertAlmostEqual(edge_metrics(self.constant, self.constant)["f1"], 1.0)

    def test_ssim_decreases_for_corruption(self):
        noisy = self.constant.copy()
        noisy[3:8, 3:8] = 240
        self.assertLess(ssim_local(self.constant, noisy), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
