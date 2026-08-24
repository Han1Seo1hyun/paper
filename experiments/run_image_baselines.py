"""Evaluate the official invisible-watermark image-domain baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ppgs_watermark.attacks import apply_attack
from ppgs_watermark.evaluation import extraction_metrics


def _to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/baselines"))
    parser.add_argument(
        "--methods", nargs="+", choices=("dwtDctSvd", "rivaGan"), default=["dwtDctSvd", "rivaGan"]
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["clean", "jpeg", "resize", "noise", "blur", "crop", "rotation", "color", "composite"],
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from imwatermark import WatermarkDecoder, WatermarkEncoder

    source_images = sorted(args.source_dir.glob("*-baseline.png"))[: args.limit]
    if not source_images:
        raise ValueError("source directory has no *-baseline.png images")

    for method in args.methods:
        payload_length = 32 if method == "rivaGan" else 256
        if method == "rivaGan":
            WatermarkEncoder.loadModel()
            WatermarkDecoder.loadModel()
        output = args.output / method.lower() / args.source_dir.name
        output.mkdir(parents=True, exist_ok=True)
        (output / "run.json").write_text(
            json.dumps(
                {
                    "format": "ppgs-image-baseline-run-v1",
                    "method": method,
                    "upstream": "ShieldMnt/invisible-watermark",
                    "upstream_revision": "68d0376d94a4701ed240af0841ec12e00676e325",
                    "source_dir": str(args.source_dir),
                    "source_images": len(source_images),
                    "payload_length": payload_length,
                    "attacks": args.attacks,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        records_path = output / "metrics.jsonl"
        records: list[dict[str, Any]] = []
        if args.resume and records_path.exists():
            records = [
                json.loads(line)
                for line in records_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        completed = {(item["prompt_index"], item["attack"]) for item in records}

        for path in source_images:
            prompt_index = int(path.name.split("-", 1)[0])
            if all((prompt_index, attack) in completed for attack in args.attacks):
                continue
            expected = np.random.default_rng(70_000 + prompt_index).integers(
                0, 2, payload_length, dtype=np.uint8
            )
            encoder = WatermarkEncoder()
            encoder.set_watermark("bits", expected.tolist())
            watermarked = _to_pil(encoder.encode(_to_bgr(Image.open(path)), method))
            watermarked.save(output / f"{prompt_index:04d}-watermarked.png")

            for attack_index, attack in enumerate(args.attacks):
                if (prompt_index, attack) in completed:
                    continue
                attacked = apply_attack(
                    watermarked,
                    attack,
                    seed=80_000 + prompt_index * 100 + attack_index,
                )
                attacked.save(output / f"{prompt_index:04d}-{attack}.png")
                decoder = WatermarkDecoder("bits", payload_length)
                recovered = np.asarray(decoder.decode(_to_bgr(attacked), method), dtype=np.uint8)
                records.append(
                    {
                        "method": method,
                        "prompt_index": prompt_index,
                        "attack": attack,
                        "payload_length": payload_length,
                        **extraction_metrics(expected, recovered, false_positive_rate=1e-6),
                    }
                )
                records_path.write_text(
                    "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
                    encoding="utf-8",
                )

        summary = {}
        for attack in args.attacks:
            selected = [item for item in records if item["attack"] == attack]
            if selected:
                summary[attack] = {
                    "samples": len(selected),
                    "mean_bit_accuracy": float(np.mean([item["bit_accuracy"] for item in selected])),
                    "true_positive_rate": float(np.mean([item["detected"] for item in selected])),
                }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
