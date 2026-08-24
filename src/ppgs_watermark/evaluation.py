"""Metrics for PPGS detection, traceability, and latent preservation."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .core import bit_accuracy


def normalized_inversion_error(reference: np.ndarray, recovered: np.ndarray) -> float:
    """Relative L2 error between terminal and DDIM-inverted latents."""

    expected = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(recovered, dtype=np.float64)
    if expected.shape != actual.shape:
        raise ValueError("reference and recovered latents must have the same shape")
    denominator = float(np.linalg.norm(expected.reshape(-1)))
    if denominator == 0:
        raise ValueError("reference latent norm must be non-zero")
    return float(np.linalg.norm((actual - expected).reshape(-1)) / denominator)


def gaussian_statistics(latents: np.ndarray) -> dict[str, float]:
    """Mean, standard-deviation drift, and one-sample normal KS statistic."""

    values = np.sort(np.asarray(latents, dtype=np.float64).reshape(-1))
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("latents must contain finite values")
    cdf = np.fromiter(
        (0.5 * (1.0 + math.erf(float(v) / math.sqrt(2.0))) for v in values),
        dtype=np.float64,
        count=values.size,
    )
    upper = np.arange(1, values.size + 1, dtype=np.float64) / values.size
    lower = np.arange(0, values.size, dtype=np.float64) / values.size
    ks = float(max(np.max(upper - cdf), np.max(cdf - lower)))
    # Stephens-corrected asymptotic two-sided Kolmogorov survival series.
    root_n = math.sqrt(values.size)
    scaled = (root_n + 0.12 + 0.11 / root_n) * ks
    ks_pvalue = float(
        np.clip(
            2.0
            * sum(
                ((-1.0) ** (term - 1)) * math.exp(-2.0 * term * term * scaled * scaled)
                for term in range(1, 101)
            ),
            0.0,
            1.0,
        )
    )
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "abs_mean": abs(float(values.mean())),
        "abs_std_minus_one": abs(float(values.std()) - 1.0),
        "ks_statistic": ks,
        "ks_pvalue_approx": ks_pvalue,
    }


def binomial_accuracy_threshold(payload_length: int, *, false_positive_rate: float) -> float:
    """Smallest accuracy whose random-bit binomial tail is at most the FPR."""

    if payload_length < 1:
        raise ValueError("payload_length must be positive")
    if not 0 < false_positive_rate < 1:
        raise ValueError("false_positive_rate must be in (0, 1)")
    denominator = 2**payload_length
    tail = 0
    for correct in range(payload_length, -1, -1):
        tail += math.comb(payload_length, correct)
        if tail / denominator > false_positive_rate:
            required = correct + 1
            if required > payload_length:
                raise ValueError("requested FPR is below the attainable exact-match rate")
            return required / payload_length
    return 0.0


def true_positive_rate(scores: Sequence[float], *, threshold: float) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores must contain finite values")
    return float(np.mean(values >= threshold))


def extraction_metrics(
    expected: Sequence[int], recovered: Sequence[int], *, false_positive_rate: float = 1e-6
) -> dict[str, float | bool]:
    accuracy = bit_accuracy(expected, recovered)
    threshold = binomial_accuracy_threshold(
        len(expected), false_positive_rate=false_positive_rate
    )
    return {
        "bit_accuracy": accuracy,
        "threshold": threshold,
        "detected": accuracy >= threshold,
        "target_false_positive_rate": false_positive_rate,
    }
