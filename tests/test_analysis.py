import unittest

from ppgs_watermark import (
    binomial_false_positive_rate,
    detection_curve,
    frechet_distance_from_features,
    paired_t_statistic,
    user_scale_attribution,
)


class PaperAnalysisTests(unittest.TestCase):
    def test_binomial_fpr(self):
        self.assertEqual(binomial_false_positive_rate(4, 1.0), 1 / 16)
        self.assertEqual(binomial_false_positive_rate(4, 0.0), 1.0)

    def test_detection_curve(self):
        curve = detection_curve([1.0, 0.75], 4)
        self.assertEqual(curve[-1]["true_positive_rate"], 0.5)
        self.assertEqual(curve[-1]["false_positive_rate"], 1 / 16)

    def test_paired_t_statistic(self):
        self.assertEqual(paired_t_statistic([1, 2, 3], [1, 2, 3]), 0.0)
        self.assertGreater(paired_t_statistic([1, 1, 1], [1, 2, 3]), 0)

    def test_frechet_distance_from_features(self):
        features = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
        self.assertAlmostEqual(frechet_distance_from_features(features, features), 0.0)
        self.assertGreater(
            frechet_distance_from_features(features, [[2, 3], [3, 2], [2.5, 2.5]]),
            0.0,
        )

    def test_user_scale_attribution(self):
        result = user_scale_attribution(
            [1, 0] * 16, [1, 0] * 16, [10, 100], seed=4
        )
        self.assertTrue(all(item["correct_attribution"] for item in result))


if __name__ == "__main__":
    unittest.main()
