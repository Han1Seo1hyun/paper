"""Soft-guidance schedules from Algorithm 2 and the ablation study."""

from __future__ import annotations

import math
from typing import Literal

GuidanceSchedule = Literal[
    "exponential", "paper_exponential", "linear", "cosine"
]


def guidance_scale(
    step_index: int,
    total_steps: int,
    *,
    maximum: float = 7.5,
    decay: float = 2.0,
    minimum: float | None = None,
    schedule: GuidanceSchedule = "exponential",
) -> float:
    """Return CFG strength for a denoising step ordered T, ..., 1.

    ``paper_exponential`` exactly follows equation (25), including its
    denominator ``T``. The paper's Table 2 instead specifies ``g_min=0.1``
    and the surrounding text says the last step should be nearly
    unconditional. The default ``exponential`` mode therefore interpolates a
    normalized exponential curve exactly from ``maximum`` to ``minimum``.
    Linear and cosine modes use the same exact endpoints for the ablation.
    """

    if total_steps < 1 or not 0 <= step_index < total_steps:
        raise ValueError("step_index must be in [0, total_steps)")
    if maximum < 0 or decay < 0 or (minimum is not None and minimum < 0):
        raise ValueError("guidance parameters must be non-negative")
    paper_progress = step_index / total_steps  # (T-t)/T in equation (25)
    progress = 0.0 if total_steps == 1 else step_index / (total_steps - 1)
    floor = 0.0 if minimum is None else minimum
    if floor > maximum:
        raise ValueError("minimum cannot exceed maximum")
    if schedule == "paper_exponential":
        value = maximum * math.exp(-decay * paper_progress)
        return max(floor, value)
    if schedule == "exponential":
        if decay == 0:
            return maximum + (floor - maximum) * progress
        endpoint = math.exp(-decay)
        weight = (math.exp(-decay * progress) - endpoint) / (1.0 - endpoint)
        return floor + (maximum - floor) * weight
    if schedule == "linear":
        return maximum + (floor - maximum) * progress
    if schedule == "cosine":
        weight = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (maximum - floor) * weight
    raise ValueError(f"unknown schedule: {schedule}")
