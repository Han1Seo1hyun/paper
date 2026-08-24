"""Proportion-Preserved Gaussian Shading (PPGS)."""

from .core import (
    EmbeddingMetadata,
    bit_accuracy,
    decode_latents,
    embed_watermark,
    symbol_probabilities,
)
from .guidance import guidance_scale
from .evaluation import (
    binomial_accuracy_threshold,
    extraction_metrics,
    gaussian_statistics,
    normalized_inversion_error,
)
from .analysis import (
    binomial_false_positive_rate,
    detection_curve,
    frechet_distance_from_features,
    paired_t_statistic,
    user_scale_attribution,
)
from .baselines import (
    GaussianShadingSecret,
    TreeRingSecret,
    gaussian_shading_decode,
    gaussian_shading_embed,
    tree_ring_distance,
    tree_ring_embed,
    tree_ring_p_value,
)

__all__ = [
    "EmbeddingMetadata",
    "bit_accuracy",
    "decode_latents",
    "embed_watermark",
    "guidance_scale",
    "binomial_accuracy_threshold",
    "extraction_metrics",
    "gaussian_statistics",
    "normalized_inversion_error",
    "binomial_false_positive_rate",
    "detection_curve",
    "frechet_distance_from_features",
    "paired_t_statistic",
    "user_scale_attribution",
    "GaussianShadingSecret",
    "TreeRingSecret",
    "gaussian_shading_decode",
    "gaussian_shading_embed",
    "tree_ring_distance",
    "tree_ring_embed",
    "tree_ring_p_value",
    "symbol_probabilities",
]
