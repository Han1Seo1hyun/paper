"""Reproduce Table 3's Gaussian-prior checks for fixed payload proportions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ppgs_watermark.core import embed_watermark
from ppgs_watermark.evaluation import gaussian_statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("outputs/paper/distribution.json"))
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("samples must be positive")

    report = {"format": "ppgs-distribution-v1", "samples_per_gamma": args.samples, "gamma": {}}
    for gamma in (0.1, 0.3, 0.5, 0.7, 0.9):
        latents = []
        one_count = round(256 * gamma)
        for index in range(args.samples):
            rng = np.random.default_rng(120_000 + index)
            payload = np.concatenate(
                [np.ones(one_count, dtype=np.uint8), np.zeros(256 - one_count, dtype=np.uint8)]
            )
            rng.shuffle(payload)
            latent, _ = embed_watermark(
                payload,
                (1, 4, 64, 64),
                payload_layout="spatial_tile",
                spatial_copies=(1, 8, 8),
                public_seed=2026,
                sampling_seed=130_000 + index,
            )
            latents.append(latent)
        actual_gamma = one_count / 256
        report["gamma"][f"{gamma:.1f}"] = {
            "requested_gamma": gamma,
            "realized_gamma": actual_gamma,
            **gaussian_statistics(np.stack(latents)),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
