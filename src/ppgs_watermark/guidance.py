"""Soft-guidance schedules from Algorithm 2 and the ablation study."""

from __future__ import annotations

import math
from typing import Literal


def guidance_scale(
    step_index: int,
    total_steps: int,
    *,
    maximum: float = 7.5,
    decay: float = 2.0,
    minimum: float | None = None,
    schedule: Literal["exponential", "linear", "cosine"] = "exponential",
) -> float:
    """Return CFG strength for a denoising step ordered T, ..., 1.

    Exponential mode exactly follows equation (25). ``minimum`` is optional
    because Table 2 names g_min=0.1 while equation (25) does not include it.
    Linear and cosine modes are supplied for reproducing the paper's ablation.
    """

    if total_steps < 1 or not 0 <= step_index < total_steps:
        raise ValueError("step_index must be in [0, total_steps)")
    if maximum < 0 or decay < 0 or (minimum is not None and minimum < 0):
        raise ValueError("guidance parameters must be non-negative")
    progress = step_index / total_steps  # (T-t)/T in Algorithm 2
    floor = 0.0 if minimum is None else minimum
    if schedule == "exponential":
        value = maximum * math.exp(-decay * progress)
        return max(floor, value)
    if schedule == "linear":
        return maximum + (floor - maximum) * progress
    if schedule == "cosine":
        weight = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (maximum - floor) * weight
    raise ValueError(f"unknown schedule: {schedule}")
