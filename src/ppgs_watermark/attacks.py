"""Image perturbations used by the paper's clean/adversarial protocol.

All geometric attacks return the original image size so that Algorithm 3
always inverts to the same 4x64x64 latent grid at 512x512 resolution.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Literal

import numpy as np

AttackName = Literal[
    "clean",
    "jpeg",
    "resize",
    "noise",
    "blur",
    "crop",
    "rotation",
    "color",
    "composite",
]


def _pil() -> tuple[Any, Any, Any, Any]:
    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "image attacks require Pillow; install ppgs-watermark[evaluation]"
        ) from exc
    return Image, ImageEnhance, ImageFilter, ImageOps


def _rgb(image: Any) -> Any:
    Image, _, _, _ = _pil()
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    return image.convert("RGB")


def jpeg_compression(image: Any, *, quality: int = 50) -> Any:
    """JPEG round trip with the quality factor specified in Section 4.1."""

    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be in [1, 100]")
    image = _rgb(image)
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=quality)
    stream.seek(0)
    Image, _, _, _ = _pil()
    with Image.open(stream) as decoded:
        return decoded.convert("RGB").copy()


def random_resize(
    image: Any, *, minimum_scale: float = 0.75, maximum_scale: float = 1.25, seed: int = 0
) -> Any:
    """Randomly rescale, then restore the original dimensions."""

    if not 0 < minimum_scale <= maximum_scale:
        raise ValueError("resize scales must be positive and ordered")
    image = _rgb(image)
    Image, _, _, _ = _pil()
    rng = np.random.default_rng(seed)
    scale = float(rng.uniform(minimum_scale, maximum_scale))
    size = tuple(max(1, round(value * scale)) for value in image.size)
    resized = image.resize(size, Image.Resampling.BICUBIC)
    return resized.resize(image.size, Image.Resampling.BICUBIC)


def gaussian_noise(image: Any, *, sigma: float = 0.05, seed: int = 0) -> Any:
    """Add Gaussian noise where sigma is measured on the [0, 1] range."""

    if sigma < 0:
        raise ValueError("noise sigma must be non-negative")
    source = np.asarray(_rgb(image), dtype=np.float32) / 255.0
    noise = np.random.default_rng(seed).normal(0.0, sigma, size=source.shape)
    result = np.clip(source + noise, 0.0, 1.0)
    Image, _, _, _ = _pil()
    return Image.fromarray(np.rint(result * 255.0).astype(np.uint8), mode="RGB")


def gaussian_blur(image: Any, *, radius: float = 1.0) -> Any:
    if radius < 0:
        raise ValueError("blur radius must be non-negative")
    _, _, ImageFilter, _ = _pil()
    return _rgb(image).filter(ImageFilter.GaussianBlur(radius=radius))


def random_crop(
    image: Any, *, maximum_area_removal: float = 0.10, seed: int = 0
) -> Any:
    """Remove up to 10% of area and resize back, matching Section 4.1."""

    if not 0 <= maximum_area_removal < 1:
        raise ValueError("maximum_area_removal must be in [0, 1)")
    image = _rgb(image)
    Image, _, _, _ = _pil()
    rng = np.random.default_rng(seed)
    retained_area = 1.0 - float(rng.uniform(0.0, maximum_area_removal))
    side_scale = retained_area**0.5
    crop_width = max(1, round(image.width * side_scale))
    crop_height = max(1, round(image.height * side_scale))
    left = int(rng.integers(0, image.width - crop_width + 1))
    top = int(rng.integers(0, image.height - crop_height + 1))
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize(image.size, Image.Resampling.BICUBIC)


def small_rotation(image: Any, *, maximum_degrees: float = 15.0, seed: int = 0) -> Any:
    if maximum_degrees < 0:
        raise ValueError("maximum_degrees must be non-negative")
    image = _rgb(image)
    Image, _, _, _ = _pil()
    angle = float(np.random.default_rng(seed).uniform(-maximum_degrees, maximum_degrees))
    # Reflect padding avoids a large constant-color corner signal. Center-crop
    # back to the original dimensions after rotation.
    diagonal = int(np.ceil((image.width**2 + image.height**2) ** 0.5))
    pad_x = max(0, (diagonal - image.width + 1) // 2)
    pad_y = max(0, (diagonal - image.height + 1) // 2)
    padded_values = np.pad(
        np.asarray(image),
        ((pad_y, pad_y), (pad_x, pad_x), (0, 0)),
        mode="reflect",
    )
    padded = Image.fromarray(padded_values, mode="RGB")
    rotated = padded.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    left = (rotated.width - image.width) // 2
    top = (rotated.height - image.height) // 2
    return rotated.crop((left, top, left + image.width, top + image.height))


def color_shift(
    image: Any, *, brightness: float = 1.10, contrast: float = 1.10
) -> Any:
    if brightness <= 0 or contrast <= 0:
        raise ValueError("brightness and contrast factors must be positive")
    _, ImageEnhance, _, _ = _pil()
    result = ImageEnhance.Brightness(_rgb(image)).enhance(brightness)
    return ImageEnhance.Contrast(result).enhance(contrast)


def apply_attack(image: Any, name: AttackName, *, seed: int = 0) -> Any:
    """Apply a named paper attack with reproducible default strength."""

    if name == "clean":
        return _rgb(image).copy()
    if name == "jpeg":
        return jpeg_compression(image, quality=50)
    if name == "resize":
        return random_resize(image, seed=seed)
    if name == "noise":
        return gaussian_noise(image, sigma=0.05, seed=seed)
    if name == "blur":
        return gaussian_blur(image, radius=1.0)
    if name == "crop":
        return random_crop(image, maximum_area_removal=0.10, seed=seed)
    if name == "rotation":
        return small_rotation(image, maximum_degrees=15.0, seed=seed)
    if name == "color":
        return color_shift(image)
    if name == "composite":
        result = jpeg_compression(image, quality=50)
        result = random_resize(result, seed=seed)
        result = gaussian_noise(result, sigma=0.05, seed=seed + 1)
        result = gaussian_blur(result, radius=1.0)
        result = random_crop(result, maximum_area_removal=0.10, seed=seed + 2)
        result = small_rotation(result, maximum_degrees=15.0, seed=seed + 3)
        return color_shift(result)
    raise ValueError(f"unknown attack: {name}")
