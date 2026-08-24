"""Paper-faithful PPGS latent sampling and decoding.

The implementation follows Algorithms 1 and 3 and equations (12)-(23).
It deliberately uses NumPy so that the statistically important part can be
tested without downloading a diffusion model or installing PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Literal, Mapping, Sequence

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
    payload_layout: Literal["full", "repeat", "spatial_tile"] = "full"
    channel_copies: int = 1
    height_copies: int = 1
    width_copies: int = 1

    @property
    def repetition_count(self) -> int:
        if self.payload_layout == "spatial_tile":
            return self.channel_copies * self.height_copies * self.width_copies
        return self.capacity_bits // self.payload_length

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable public extraction manifest."""

        return {
            "format": "ppgs-embedding-v1",
            "latent_shape": list(self.latent_shape),
            "bits_per_position": self.bits_per_position,
            "payload_length": self.payload_length,
            "capacity_bits": self.capacity_bits,
            "gamma": self.gamma,
            "public_seed": self.public_seed,
            "payload_layout": self.payload_layout,
            "channel_copies": self.channel_copies,
            "height_copies": self.height_copies,
            "width_copies": self.width_copies,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "EmbeddingMetadata":
        if values.get("format", "ppgs-embedding-v1") != "ppgs-embedding-v1":
            raise ValueError("unsupported PPGS metadata format")
        shape = values.get("latent_shape")
        if not isinstance(shape, (list, tuple)):
            raise ValueError("latent_shape is missing from PPGS metadata")
        layout = str(values.get("payload_layout", "full"))
        if layout not in {"full", "repeat", "spatial_tile"}:
            raise ValueError(f"unknown payload layout: {layout}")
        return cls(
            latent_shape=tuple(int(v) for v in shape),
            bits_per_position=int(values["bits_per_position"]),
            payload_length=int(values["payload_length"]),
            capacity_bits=int(values["capacity_bits"]),
            gamma=float(values["gamma"]),
            public_seed=int(values["public_seed"]),
            payload_layout=layout,  # type: ignore[arg-type]
            channel_copies=int(values.get("channel_copies", 1)),
            height_copies=int(values.get("height_copies", 1)),
            width_copies=int(values.get("width_copies", 1)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingMetadata":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("PPGS metadata must be a JSON object")
        return cls.from_dict(values)


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
    payload_layout: Literal["full", "repeat", "spatial_tile"] = "full",
    spatial_copies: tuple[int, int, int] = (1, 8, 8),
    repeat_payload: bool | None = None,
    dtype: np.dtype = np.dtype("float32"),
) -> tuple[NDArray[np.floating], EmbeddingMetadata]:
    """Embed bits by proportion-aware inverse-CDF sampling (Algorithm 1).

    The paper's algorithm requires one m-bit symbol per latent position. Its
    experiments also report a 256-bit payload in a 4x64x64 latent. Use
    ``payload_layout="spatial_tile"`` for the Gaussian-Shading experimental
    interpretation: a 4x8x8 payload is tiled 8x8 across a 4x64x64 latent and
    extraction applies block-wise majority voting. ``repeat`` preserves the
    earlier flat repeat interpretation.
    ``repeat_payload`` remains as a compatibility alias.
    """

    bits = _as_bits(watermark)
    shape = tuple(int(v) for v in latent_shape)
    if not shape or any(v <= 0 for v in shape):
        raise ValueError("latent_shape must contain positive dimensions")
    if bits_per_position < 1:
        raise ValueError("bits_per_position must be positive")

    positions = int(np.prod(shape))
    capacity = positions * bits_per_position
    if repeat_payload is not None:
        requested = "repeat" if repeat_payload else "full"
        if payload_layout != "full" and payload_layout != requested:
            raise ValueError("payload_layout and repeat_payload disagree")
        payload_layout = requested
    if payload_layout not in {"full", "repeat", "spatial_tile"}:
        raise ValueError(f"unknown payload layout: {payload_layout}")

    copy_factors = tuple(int(value) for value in spatial_copies)
    if len(copy_factors) != 3 or any(value < 1 for value in copy_factors):
        raise ValueError("spatial_copies must contain three positive integers")

    if payload_layout == "spatial_tile":
        if len(shape) not in {3, 4} or (len(shape) == 4 and shape[0] != 1):
            raise ValueError("spatial_tile requires a CxHxW or 1xCxHxW latent")
        channels, height, width = shape[-3:]
        if any(size % copies for size, copies in zip((channels, height, width), copy_factors)):
            raise ValueError("spatial copy factors must divide C, H, and W")
        base_shape = tuple(
            size // copies for size, copies in zip((channels, height, width), copy_factors)
        )
        expected_payload = int(np.prod(base_shape)) * bits_per_position
        if bits.size != expected_payload:
            raise ValueError(
                f"spatial_tile expects {expected_payload} bits for base grid "
                f"{base_shape}, got {bits.size}"
            )
        base_symbols = _bits_to_symbols(bits, bits_per_position).reshape(base_shape)
        tiled_symbols = np.tile(base_symbols, copy_factors).reshape(-1)
        expanded = _symbols_to_bits(tiled_symbols, bits_per_position)
        permutation = _public_permutation(capacity, public_seed)
        permuted = expanded[permutation]
        symbols = _bits_to_symbols(permuted, bits_per_position)
    elif payload_layout == "repeat":
        if capacity % bits.size:
            raise ValueError("payload length must divide latent capacity when repeated")
        expanded = np.tile(bits, capacity // bits.size)
    else:
        if bits.size != capacity:
            raise ValueError(
                f"watermark has {bits.size} bits, but latent capacity is {capacity}; "
                "use payload_layout='repeat' when the payload divides the capacity"
            )
        expanded = bits.copy()

    # Equation (12) is defined on the complete sequence mapped into W. The
    # repeated 256-bit experimental interpretation has the same mean, but
    # using expanded here also keeps the definition correct for future layouts.
    gamma = float(expanded.mean())
    if payload_layout != "spatial_tile":
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
        payload_layout=payload_layout,
        channel_copies=copy_factors[0] if payload_layout == "spatial_tile" else 1,
        height_copies=copy_factors[1] if payload_layout == "spatial_tile" else 1,
        width_copies=copy_factors[2] if payload_layout == "spatial_tile" else 1,
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

    if metadata.payload_layout == "spatial_tile":
        if len(metadata.latent_shape) not in {3, 4}:
            raise ValueError("spatial_tile metadata requires CxHxW latent dimensions")
        channels, height, width = metadata.latent_shape[-3:]
        factors = (
            metadata.channel_copies,
            metadata.height_copies,
            metadata.width_copies,
        )
        if any(size % copies for size, copies in zip((channels, height, width), factors)):
            raise ValueError("metadata spatial copy factors do not divide latent shape")
        base_shape = tuple(
            size // copies for size, copies in zip((channels, height, width), factors)
        )
        permutation = _public_permutation(metadata.capacity_bits, metadata.public_seed)
        expanded = np.empty(metadata.capacity_bits, dtype=np.uint8)
        expanded[permutation] = permuted
        bit_grid = expanded.reshape(
            channels, height, width, metadata.bits_per_position
        )
        copies = []
        for channel_index in range(factors[0]):
            for height_index in range(factors[1]):
                for width_index in range(factors[2]):
                    copies.append(
                        bit_grid[
                            channel_index * base_shape[0] : (channel_index + 1)
                            * base_shape[0],
                            height_index * base_shape[1] : (height_index + 1)
                            * base_shape[1],
                            width_index * base_shape[2] : (width_index + 1)
                            * base_shape[2],
                            :,
                        ]
                    )
        votes = np.stack(copies).sum(axis=0)
        tiled_bits = (votes * 2 >= len(copies)).astype(np.uint8).reshape(-1)
        return tiled_bits

    permutation = _public_permutation(metadata.capacity_bits, metadata.public_seed)
    expanded = np.empty(metadata.capacity_bits, dtype=np.uint8)
    expanded[permutation] = permuted
    if metadata.payload_layout == "full":
        return expanded
    if metadata.capacity_bits % metadata.payload_length:
        raise ValueError("repeated payload length does not divide latent capacity")
    votes = expanded.reshape(metadata.repetition_count, metadata.payload_length).sum(axis=0)
    return (votes * 2 >= metadata.repetition_count).astype(np.uint8)


def bit_accuracy(expected: Sequence[int], actual: Sequence[int]) -> float:
    lhs = _as_bits(expected)
    rhs = _as_bits(actual)
    if lhs.shape != rhs.shape:
        raise ValueError("bit sequences must have the same length")
    return float(np.mean(lhs == rhs))
