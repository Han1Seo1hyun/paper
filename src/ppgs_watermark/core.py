"""Paper-faithful PPGS latent sampling and decoding.

The implementation follows Algorithms 1 and 3 and equations (12)-(23).
It deliberately uses NumPy so that the statistically important part can be
tested without downloading a diffusion model or installing PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

_NORMAL = NormalDist()


@dataclass(frozen=True)
class EmbeddingMetadata:
    """Public information required to reverse PPGS sampling.

    ``gamma`` and ``public_seed`` are not secret keys. The paper's decoder
    needs the same interval partition and public permutation as the encoder.
    """

    latent_shape: tuple[int, ...]
    bits_per_position: int
    payload_length: int
    capacity_bits: int
    gamma: float
    public_seed: int

    @property
    def repetition_count(self) -> int:
        return self.capacity_bits // self.payload_length


def _as_bits(bits: Iterable[int]) -> NDArray[np.uint8]:
    raw = np.asarray(list(bits)).reshape(-1)
    if raw.size == 0:
        raise ValueError("watermark must contain at least one bit")
    if not np.all((raw == 0) | (raw == 1)):
        raise ValueError("watermark values must be binary (0 or 1)")
    return raw.astype(np.uint8, copy=False)


def _public_permutation(length: int, seed: int) -> NDArray[np.int64]:
    # Pin PCG64 rather than relying on default_rng's current default so the
    # public mapping is reproducible across future NumPy releases.
    return np.random.Generator(np.random.PCG64(seed)).permutation(length)


def symbol_probabilities(bits_per_position: int, gamma: float) -> NDArray[np.float64]:
    """Return the Bernoulli-derived symbol probabilities from equation (14).

    The exponent printed as ``l-k_i`` in the paper is dimensionally
    inconsistent for an m-bit symbol. This implementation uses ``m-k_i``,
    which normalizes the 2**m symbol probabilities to one.
    """

    if bits_per_position < 1:
        raise ValueError("bits_per_position must be positive")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    count = 1 << bits_per_position
    weights = np.fromiter(
        (int(i).bit_count() for i in range(count)), dtype=np.int64, count=count
    )
    probabilities = (gamma**weights) * ((1.0 - gamma) ** (bits_per_position - weights))
    probabilities /= probabilities.sum()
    return probabilities.astype(np.float64, copy=False)


def _partition(bits_per_position: int, gamma: float) -> NDArray[np.float64]:
    probabilities = symbol_probabilities(bits_per_position, gamma)
    boundaries = np.concatenate(([0.0], np.cumsum(probabilities)))
    boundaries[-1] = 1.0
    return boundaries


def _bits_to_symbols(bits: NDArray[np.uint8], width: int) -> NDArray[np.int64]:
    grouped = bits.reshape(-1, width).astype(np.int64)
    powers = 1 << np.arange(width - 1, -1, -1, dtype=np.int64)
    return grouped @ powers


def _symbols_to_bits(symbols: NDArray[np.int64], width: int) -> NDArray[np.uint8]:
    shifts = np.arange(width - 1, -1, -1, dtype=np.int64)
    return ((symbols[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)


def _normal_ppf(values: NDArray[np.float64]) -> NDArray[np.float64]:
    flat = np.fromiter((_NORMAL.inv_cdf(float(v)) for v in values), dtype=np.float64)
    return flat.reshape(values.shape)


def _normal_cdf(values: NDArray[np.float64]) -> NDArray[np.float64]:
    flat = np.fromiter((_NORMAL.cdf(float(v)) for v in values), dtype=np.float64)
    return flat.reshape(values.shape)


def embed_watermark(
    watermark: Sequence[int] | NDArray[np.integer],
    latent_shape: Sequence[int],
    *,
    bits_per_position: int = 1,
    public_seed: int = 2026,
    sampling_seed: int | None = None,
    repeat_payload: bool = False,
    dtype: np.dtype = np.dtype("float32"),
) -> tuple[NDArray[np.floating], EmbeddingMetadata]:
    """Embed bits by proportion-aware inverse-CDF sampling (Algorithm 1).

    The paper's algorithm requires one m-bit symbol per latent position. Its
    experiments also report a 256-bit payload in a 4x64x64 latent. Set
    ``repeat_payload=True`` for that experimental interpretation: the payload
    is tiled to capacity and extraction applies majority voting.
    """

    bits = _as_bits(watermark)
    shape = tuple(int(v) for v in latent_shape)
    if not shape or any(v <= 0 for v in shape):
        raise ValueError("latent_shape must contain positive dimensions")
    if bits_per_position < 1:
        raise ValueError("bits_per_position must be positive")

    positions = int(np.prod(shape))
    capacity = positions * bits_per_position
    if repeat_payload:
        if capacity % bits.size:
            raise ValueError("payload length must divide latent capacity when repeated")
        expanded = np.tile(bits, capacity // bits.size)
    else:
        if bits.size != capacity:
            raise ValueError(
                f"watermark has {bits.size} bits, but latent capacity is {capacity}; "
                "pass repeat_payload=True when the payload divides the capacity"
            )
        expanded = bits.copy()

    gamma = float(bits.mean())
    permutation = _public_permutation(capacity, public_seed)
    permuted = expanded[permutation]
    symbols = _bits_to_symbols(permuted, bits_per_position)

    boundaries = _partition(bits_per_position, gamma)
    lower = boundaries[symbols]
    widths = boundaries[symbols + 1] - lower
    if np.any(widths <= 0):
        raise ValueError("watermark contains a symbol with zero probability under gamma")

    rng = np.random.default_rng(sampling_seed)
    uniforms = lower + widths * rng.random(positions)
    # Avoid infinities if a random backend ever produces an exact endpoint.
    uniforms = np.clip(uniforms, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    latents = _normal_ppf(uniforms).astype(dtype, copy=False).reshape(shape)
    metadata = EmbeddingMetadata(
        latent_shape=shape,
        bits_per_position=bits_per_position,
        payload_length=int(bits.size),
        capacity_bits=capacity,
        gamma=gamma,
        public_seed=public_seed,
    )
    return latents, metadata


def decode_latents(
    latents: NDArray[np.floating], metadata: EmbeddingMetadata
) -> NDArray[np.uint8]:
    """Reverse PPGS intervals, the public permutation, and repetitions."""

    values = np.asarray(latents, dtype=np.float64)
    if values.shape != metadata.latent_shape:
        raise ValueError(f"expected latent shape {metadata.latent_shape}, got {values.shape}")
    boundaries = _partition(metadata.bits_per_position, metadata.gamma)
    uniforms = _normal_cdf(values.reshape(-1))
    symbols = np.searchsorted(boundaries[1:], uniforms, side="right")
    symbols = np.minimum(symbols, (1 << metadata.bits_per_position) - 1)
    permuted = _symbols_to_bits(symbols.astype(np.int64), metadata.bits_per_position)

    permutation = _public_permutation(metadata.capacity_bits, metadata.public_seed)
    expanded = np.empty(metadata.capacity_bits, dtype=np.uint8)
    expanded[permutation] = permuted
    if metadata.repetition_count == 1:
        return expanded
    votes = expanded.reshape(metadata.repetition_count, metadata.payload_length).sum(axis=0)
    return (votes * 2 >= metadata.repetition_count).astype(np.uint8)


def bit_accuracy(expected: Sequence[int], actual: Sequence[int]) -> float:
    lhs = _as_bits(expected)
    rhs = _as_bits(actual)
    if lhs.shape != rhs.shape:
        raise ValueError("bit sequences must have the same length")
    return float(np.mean(lhs == rhs))
