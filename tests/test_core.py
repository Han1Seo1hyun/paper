import unittest

import numpy as np

from ppgs_watermark import (
    bit_accuracy,
    decode_latents,
    embed_watermark,
    guidance_scale,
    symbol_probabilities,
)


class PPGSCoreTests(unittest.TestCase):
    def test_exact_round_trip_one_bit(self):
        bits = np.array([0, 1, 1, 0] * 4, dtype=np.uint8)
        latents, metadata = embed_watermark(bits, (4, 2, 2), sampling_seed=7)
        np.testing.assert_array_equal(decode_latents(latents, metadata), bits)

    def test_exact_round_trip_two_bits(self):
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1] * 2, dtype=np.uint8)
        latents, metadata = embed_watermark(
            bits, (2, 2, 2), bits_per_position=2, sampling_seed=9
        )
        np.testing.assert_array_equal(decode_latents(latents, metadata), bits)

    def test_256_bit_payload_repeats_to_sd_latent(self):
        bits = np.random.default_rng(3).integers(0, 2, 256, dtype=np.uint8)
        latents, metadata = embed_watermark(
            bits, (4, 64, 64), repeat_payload=True, sampling_seed=11
        )
        recovered = decode_latents(latents, metadata)
        self.assertEqual(metadata.repetition_count, 64)
        self.assertEqual(bit_accuracy(bits, recovered), 1.0)

    def test_repetition_recovers_from_sparse_latent_errors(self):
        bits = np.random.default_rng(13).integers(0, 2, 16, dtype=np.uint8)
        latents, metadata = embed_watermark(
            bits, (1024,), repeat_payload=True, sampling_seed=17
        )
        damaged = latents.copy()
        damaged[:20] *= -1
        self.assertGreaterEqual(bit_accuracy(bits, decode_latents(damaged, metadata)), 0.9)

    def test_marginal_latent_distribution_is_standard_normal(self):
        # Exactly 30% ones makes the empirical m=1 symbol frequencies match
        # the interval masses used by the PPGS mixture proof in equation (20).
        bits = np.array([1] * 6000 + [0] * 14000, dtype=np.uint8)
        latents, _ = embed_watermark(bits, (20000,), sampling_seed=23)
        self.assertLess(abs(float(latents.mean())), 0.03)
        self.assertLess(abs(float(latents.std()) - 1.0), 0.03)

    def test_sampling_is_deterministic_for_both_seeds(self):
        bits = [0, 1] * 8
        first, _ = embed_watermark(bits, (16,), public_seed=4, sampling_seed=5)
        second, _ = embed_watermark(bits, (16,), public_seed=4, sampling_seed=5)
        np.testing.assert_array_equal(first, second)

    def test_symbol_probabilities_normalize(self):
        probabilities = symbol_probabilities(3, 0.3)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertTrue(np.all(probabilities >= 0))

    def test_non_binary_values_are_rejected_before_integer_cast(self):
        with self.assertRaises(ValueError):
            embed_watermark([0, 0.5, 1, 0], (4,))
        with self.assertRaises(ValueError):
            embed_watermark([0, 256, 1, 0], (4,))

    def test_exponential_soft_guidance_matches_equation_25(self):
        self.assertEqual(guidance_scale(0, 50), 7.5)
        self.assertAlmostEqual(
            guidance_scale(49, 50, maximum=7.5, decay=2.0),
            7.5 * np.exp(-2.0 * 49 / 50),
        )


if __name__ == "__main__":
    unittest.main()
