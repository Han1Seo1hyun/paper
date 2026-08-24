"""Reproducible latent primitives for the paper's diffusion baselines."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

_NORMAL = NormalDist()


def _ppf(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.fromiter(
        (_NORMAL.inv_cdf(float(value)) for value in values.reshape(-1)),
        dtype=np.float64,
        count=values.size,
    ).reshape(values.shape)


@dataclass(frozen=True)
class GaussianShadingSecret:
    """Secret XOR mask and public layout used by the keyed GS baseline."""

    key: NDArray[np.uint8]
    payload_shape: tuple[int, int, int]
    copy_factors: tuple[int, int, int]


def gaussian_shading_embed(
    payload: Sequence[int],
    latent_shape: tuple[int, int, int, int] = (1, 4, 64, 64),
    *,
    copy_factors: tuple[int, int, int] = (1, 8, 8),
    key_seed: int = 999_999,
    sampling_seed: int = 0,
) -> tuple[NDArray[np.float32], GaussianShadingSecret]:
    """Official Gaussian-Shading sign conditioning with an XOR secret key."""

    if latent_shape[0] != 1:
        raise ValueError("the baseline currently supports one latent at a time")
    channels, height, width = latent_shape[1:]
    if any(size % factor for size, factor in zip((channels, height, width), copy_factors)):
        raise ValueError("copy factors must divide the latent dimensions")
    payload_shape = tuple(
        size // factor for size, factor in zip((channels, height, width), copy_factors)
    )
    bits = np.asarray(payload, dtype=np.uint8).reshape(-1)
    if bits.size != int(np.prod(payload_shape)) or np.any(bits > 1):
        raise ValueError(f"expected {int(np.prod(payload_shape))} binary payload bits")
    expanded = np.tile(bits.reshape(payload_shape), copy_factors).reshape(latent_shape)
    key_rng = np.random.Generator(np.random.PCG64(key_seed))
    key = key_rng.integers(0, 2, latent_shape, dtype=np.uint8)
    encrypted = np.bitwise_xor(expanded, key)
    sample_rng = np.random.Generator(np.random.PCG64(sampling_seed))
    uniforms = (encrypted.astype(np.float64) + sample_rng.random(latent_shape)) / 2.0
    uniforms = np.clip(uniforms, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    latent = _ppf(uniforms).astype(np.float32)
    return latent, GaussianShadingSecret(key, payload_shape, copy_factors)


def gaussian_shading_decode(
    latent: NDArray[np.floating], secret: GaussianShadingSecret
) -> NDArray[np.uint8]:
    """Recover the GS payload by sign decoding, XOR, and block voting."""

    values = np.asarray(latent)
    if values.shape != secret.key.shape:
        raise ValueError("latent shape does not match the Gaussian-Shading key")
    decrypted = np.bitwise_xor((values > 0).astype(np.uint8), secret.key)[0]
    factors = secret.copy_factors
    base = secret.payload_shape
    copies = []
    for channel_index in range(factors[0]):
        for height_index in range(factors[1]):
            for width_index in range(factors[2]):
                copies.append(
                    decrypted[
                        channel_index * base[0] : (channel_index + 1) * base[0],
                        height_index * base[1] : (height_index + 1) * base[1],
                        width_index * base[2] : (width_index + 1) * base[2],
                    ]
                )
    votes = np.stack(copies).sum(axis=0)
    return (votes * 2 > len(copies)).astype(np.uint8).reshape(-1)


@dataclass(frozen=True)
class TreeRingSecret:
    mask: NDArray[np.bool_]
    pattern: NDArray[np.complex128]


def _circle_mask(height: int, width: int, radius: int) -> NDArray[np.bool_]:
    y, x = np.ogrid[:height, :width]
    return (y - height // 2) ** 2 + (x - width // 2) ** 2 <= radius**2


def tree_ring_embed(
    latent: NDArray[np.floating], *, seed: int = 999_999, channel: int = 0, radius: int = 10
) -> tuple[NDArray[np.float32], TreeRingSecret]:
    """Inject Tree-Ring's fixed random Fourier patch in one latent channel."""

    values = np.asarray(latent, dtype=np.float64)
    if values.ndim != 4 or values.shape[0] != 1:
        raise ValueError("Tree-Ring expects a 1xCxHxW latent")
    if not 0 <= channel < values.shape[1]:
        raise ValueError("channel is outside the latent")
    height, width = values.shape[-2:]
    mask_2d = _circle_mask(height, width, radius)
    mask = np.zeros(values.shape, dtype=np.bool_)
    mask[0, channel] = mask_2d
    rng = np.random.Generator(np.random.PCG64(seed))
    source = rng.standard_normal(values.shape)
    pattern = np.fft.fftshift(np.fft.fft2(source, axes=(-2, -1)), axes=(-2, -1))
    spectrum = np.fft.fftshift(np.fft.fft2(values, axes=(-2, -1)), axes=(-2, -1))
    spectrum[mask] = pattern[mask]
    embedded = np.fft.ifft2(
        np.fft.ifftshift(spectrum, axes=(-2, -1)), axes=(-2, -1)
    ).real
    return embedded.astype(np.float32), TreeRingSecret(mask, pattern)


def tree_ring_distance(latent: NDArray[np.floating], secret: TreeRingSecret) -> float:
    """Mean L1 Fourier distance used as Tree-Ring's detection score."""

    values = np.asarray(latent, dtype=np.float64)
    if values.shape != secret.mask.shape:
        raise ValueError("latent shape does not match the Tree-Ring mask")
    spectrum = np.fft.fftshift(np.fft.fft2(values, axes=(-2, -1)), axes=(-2, -1))
    return float(np.mean(np.abs(spectrum[secret.mask] - secret.pattern[secret.mask])))


def tree_ring_p_value(latent: NDArray[np.floating], secret: TreeRingSecret) -> float:
    """Official noncentral-chi-square Tree-Ring detection p-value."""

    try:
        from scipy.stats import ncx2
    except ImportError as exc:  # pragma: no cover - optional baseline dependency
        raise ImportError("Tree-Ring p-values require scipy") from exc
    values = np.asarray(latent, dtype=np.float64)
    if values.shape != secret.mask.shape:
        raise ValueError("latent shape does not match the Tree-Ring mask")
    spectrum = np.fft.fftshift(np.fft.fft2(values, axes=(-2, -1)), axes=(-2, -1))
    recovered = spectrum[secret.mask].reshape(-1)
    target = secret.pattern[secret.mask].reshape(-1)
    recovered_components = np.concatenate([recovered.real, recovered.imag])
    target_components = np.concatenate([target.real, target.imag])
    sigma = float(recovered_components.std(ddof=1))
    if sigma == 0:
        return 0.0 if np.array_equal(recovered_components, target_components) else 1.0
    noncentrality = float(np.sum((target_components / sigma) ** 2))
    statistic = float(np.sum(((recovered_components - target_components) / sigma) ** 2))
    return float(ncx2.cdf(statistic, df=target_components.size, nc=noncentrality))
