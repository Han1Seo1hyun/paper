import unittest

import numpy as np

from ppgs_watermark.baselines import (
    gaussian_shading_decode,
    gaussian_shading_embed,
    tree_ring_distance,
    tree_ring_embed,
    tree_ring_p_value,
)


class BaselineTests(unittest.TestCase):
    def test_gaussian_shading_exact_roundtrip(self):
        payload = np.random.default_rng(4).integers(0, 2, 256, dtype=np.uint8)
        latent, secret = gaussian_shading_embed(payload, sampling_seed=8)
        recovered = gaussian_shading_decode(latent, secret)
        np.testing.assert_array_equal(recovered, payload)

    def test_tree_ring_exact_roundtrip(self):
        latent = np.random.default_rng(2).standard_normal((1, 4, 64, 64))
        embedded, secret = tree_ring_embed(latent)
        self.assertLess(tree_ring_distance(embedded, secret), 1e-4)
        self.assertGreater(tree_ring_distance(latent, secret), 1.0)

    def test_tree_ring_p_value_separates_exact_patch(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("scipy is an optional baseline dependency")
        latent = np.random.default_rng(3).standard_normal((1, 4, 64, 64))
        embedded, secret = tree_ring_embed(latent)
        self.assertLess(tree_ring_p_value(embedded, secret), 1e-6)


if __name__ == "__main__":
    unittest.main()
