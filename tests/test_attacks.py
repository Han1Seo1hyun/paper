import unittest

import numpy as np

from ppgs_watermark.attacks import apply_attack


class PaperAttackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from PIL import Image
        except ImportError as exc:
            raise unittest.SkipTest("Pillow is not installed") from exc
        cls.image = Image.fromarray(
            np.random.default_rng(4).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        )

    def test_every_attack_preserves_size_and_mode(self):
        for name in (
            "clean",
            "jpeg",
            "resize",
            "noise",
            "blur",
            "crop",
            "rotation",
            "color",
            "composite",
        ):
            with self.subTest(name=name):
                attacked = apply_attack(self.image, name, seed=9)
                self.assertEqual(attacked.size, self.image.size)
                self.assertEqual(attacked.mode, "RGB")

    def test_stochastic_attacks_are_seeded(self):
        first = np.asarray(apply_attack(self.image, "composite", seed=10))
        second = np.asarray(apply_attack(self.image, "composite", seed=10))
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
