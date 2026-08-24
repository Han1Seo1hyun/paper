import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ppgs_watermark import (
    bit_accuracy,
    binomial_accuracy_threshold,
    decode_latents,
    EmbeddingMetadata,
    embed_watermark,
    gaussian_statistics,
    guidance_scale,
    normalized_inversion_error,
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
        self.assertEqual(metadata.payload_layout, "repeat")

    def test_256_bit_payload_spatially_tiles_like_gaussian_shading(self):
        bits = np.random.default_rng(5).integers(0, 2, 256, dtype=np.uint8)
        latents, metadata = embed_watermark(
            bits,
            (1, 4, 64, 64),
            payload_layout="spatial_tile",
            spatial_copies=(1, 8, 8),
            sampling_seed=12,
        )
        recovered = decode_latents(latents, metadata)
        self.assertEqual(metadata.repetition_count, 64)
        self.assertEqual(metadata.payload_layout, "spatial_tile")
        np.testing.assert_array_equal(recovered, bits)

    def test_spatial_tile_recovers_from_damaged_blocks(self):
        bits = np.random.default_rng(6).integers(0, 2, 256, dtype=np.uint8)
        latents, metadata = embed_watermark(
            bits,
            (1, 4, 64, 64),
            payload_layout="spatial_tile",
            sampling_seed=14,
        )
        damaged = latents.copy()
        damaged[:, :, :16, :] *= -1
        self.assertGreaterEqual(bit_accuracy(bits, decode_latents(damaged, metadata)), 0.99)

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
        self.assertEqual(guidance_scale(0, 50, schedule="paper_exponential"), 7.5)
        self.assertAlmostEqual(
            guidance_scale(
                49, 50, maximum=7.5, decay=2.0, schedule="paper_exponential"
            ),
            7.5 * np.exp(-2.0 * 49 / 50),
        )

    def test_experimental_guidance_hits_table_endpoints(self):
        self.assertEqual(guidance_scale(0, 50, minimum=0.1), 7.5)
        self.assertAlmostEqual(guidance_scale(49, 50, minimum=0.1), 0.1)

    def test_metadata_json_round_trip(self):
        bits = np.array([0, 1] * 8, dtype=np.uint8)
        _, metadata = embed_watermark(
            bits, (32,), payload_layout="repeat", sampling_seed=2
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            metadata.save(path)
            self.assertEqual(EmbeddingMetadata.load(path), metadata)

    def test_latent_metrics(self):
        values = np.random.default_rng(3).normal(size=4096)
        stats = gaussian_statistics(values)
        self.assertLess(stats["abs_mean"], 0.05)
        self.assertLess(stats["abs_std_minus_one"], 0.05)
        self.assertEqual(normalized_inversion_error(values, values), 0.0)

    def test_binomial_threshold_meets_fpr(self):
        threshold = binomial_accuracy_threshold(256, false_positive_rate=1e-6)
        self.assertGreater(threshold, 0.5)
        self.assertLess(threshold, 1.0)


if __name__ == "__main__":
    unittest.main()
