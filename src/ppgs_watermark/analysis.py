"""Statistical analyses used by the paper's tables and detection curves."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def binomial_false_positive_rate(payload_length: int, threshold: float) -> float:
    """Exact random-bit probability of accuracy greater than or equal to threshold."""

    if payload_length < 1:
        raise ValueError("payload_length must be positive")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    required = math.ceil(threshold * payload_length)
    numerator = sum(math.comb(payload_length, k) for k in range(required, payload_length + 1))
    return numerator / (2**payload_length)


def detection_curve(
    positive_scores: Sequence[float], payload_length: int
) -> list[dict[str, float]]:
    """Return every attainable theoretical FPR and empirical TPR point."""

    scores = np.asarray(positive_scores, dtype=np.float64)
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("positive_scores must contain finite values")
    curve = []
    for correct in range(payload_length // 2, payload_length + 1):
        threshold = correct / payload_length
        curve.append(
            {
                "threshold": threshold,
                "false_positive_rate": binomial_false_positive_rate(
                    payload_length, threshold
                ),
                "true_positive_rate": float(np.mean(scores >= threshold)),
            }
        )
    return curve


def paired_t_statistic(reference: Iterable[float], treatment: Iterable[float]) -> float:
    """Paired t statistic, matching the value reported in Tables 1 and 5."""

    lhs = np.asarray(list(reference), dtype=np.float64)
    rhs = np.asarray(list(treatment), dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.size < 2:
        raise ValueError("paired samples must have the same shape and at least two values")
    differences = rhs - lhs
    standard_deviation = float(differences.std(ddof=1))
    if standard_deviation == 0:
        return 0.0 if float(differences.mean()) == 0 else math.inf
    return float(differences.mean() / (standard_deviation / math.sqrt(differences.size)))


def frechet_distance_from_features(
    reference: Sequence[Sequence[float]] | np.ndarray,
    generated: Sequence[Sequence[float]] | np.ndarray,
) -> float:
    """Sample Fréchet distance without feature-width covariance matrices.

    The covariance trace term is evaluated through singular values of the
    centered sample cross-product. This is exact and practical for the paper's
    small batches with 2048-dimensional Inception features.
    """

    lhs = np.asarray(reference, dtype=np.float64)
    rhs = np.asarray(generated, dtype=np.float64)
    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[1] != rhs.shape[1]:
        raise ValueError("FID features must be two matrices with equal feature width")
    if len(lhs) < 2 or len(rhs) < 2:
        raise ValueError("FID requires at least two samples in each set")
    mean_difference = lhs.mean(axis=0) - rhs.mean(axis=0)
    centered_lhs = lhs - lhs.mean(axis=0)
    centered_rhs = rhs - rhs.mean(axis=0)
    lhs_trace = float(np.sum(centered_lhs**2) / (len(lhs) - 1))
    rhs_trace = float(np.sum(centered_rhs**2) / (len(rhs) - 1))
    cross = centered_lhs @ centered_rhs.T
    cross /= math.sqrt((len(lhs) - 1) * (len(rhs) - 1))
    covariance_root_trace = float(np.linalg.svd(cross, compute_uv=False).sum())
    value = float(
        mean_difference @ mean_difference
        + lhs_trace
        + rhs_trace
        - 2.0 * covariance_root_trace
    )
    return max(value, 0.0)


def user_scale_attribution(
    recovered: Sequence[int],
    expected: Sequence[int],
    user_counts: Sequence[int],
    *,
    seed: int = 0,
    chunk_size: int = 100_000,
) -> list[dict[str, float | int | bool]]:
    """Nearest-codeword attribution as the number of public user IDs grows."""

    recovered_bits = np.asarray(recovered, dtype=np.uint8).reshape(-1)
    expected_bits = np.asarray(expected, dtype=np.uint8).reshape(-1)
    if recovered_bits.shape != expected_bits.shape:
        raise ValueError("recovered and expected payloads must have the same shape")
    if not np.all((recovered_bits <= 1) & (expected_bits <= 1)):
        raise ValueError("payloads must be binary")
    true_score = float(np.mean(recovered_bits == expected_bits))
    results = []
    for count in user_counts:
        if count < 1:
            raise ValueError("user counts must be positive")
        rng = np.random.default_rng(seed + count)
        best_impostor = -1.0
        remaining = count - 1
        while remaining:
            size = min(chunk_size, remaining)
            candidates = rng.integers(
                0, 2, (size, recovered_bits.size), dtype=np.uint8
            )
            best_impostor = max(
                best_impostor,
                float(np.max(np.mean(candidates == recovered_bits, axis=1))),
            )
            remaining -= size
        results.append(
            {
                "users": count,
                "true_bit_accuracy": true_score,
                "best_impostor_accuracy": best_impostor,
                "correct_attribution": true_score > best_impostor,
            }
        )
    return results
