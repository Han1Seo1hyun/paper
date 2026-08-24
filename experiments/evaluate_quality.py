"""Compute paper-style FID batches, CLIP score, and paired t statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from ppgs_watermark.analysis import frechet_distance_from_features, paired_t_statistic


def _images(directory: Path, suffix: str) -> list[Path]:
    return sorted(directory.glob(f"*-{suffix}.png"))


def _batched(values: list[Any], size: int) -> list[list[Any]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _extract_features(
    paths: list[Path],
    preprocess: Callable[[Image.Image], Any],
    encode: Callable[[Any], Any],
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    import torch

    outputs = []
    for batch in _batched(paths, batch_size):
        tensors = torch.stack(
            [preprocess(Image.open(path).convert("RGB")) for path in batch]
        ).to(device)
        with torch.inference_mode():
            features = encode(tensors)
        outputs.append(features.detach().float().cpu().numpy())
    return np.concatenate(outputs)


def _batch_fids(
    reference: np.ndarray, generated: np.ndarray, batch_size: int
) -> list[float]:
    count = min(len(reference), len(generated))
    values = []
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        if stop - start >= 2:
            values.append(
                frechet_distance_from_features(
                    reference[start:stop], generated[start:stop]
                )
            )
    return values


def _mean_std(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--clip-cache-dir", type=Path, default=Path("work/openclip"))
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch
    from torchvision.models import Inception_V3_Weights, inception_v3

    watermarked_paths = _images(args.run_dir, "watermarked")
    baseline_paths = _images(args.run_dir, "baseline")
    if not watermarked_paths or not baseline_paths:
        raise ValueError("run directory must contain paired baseline and watermarked images")
    if len(watermarked_paths) != len(baseline_paths):
        raise ValueError("baseline and watermarked image counts differ")

    if args.reference_dir:
        reference_paths = sorted(
            path
            for path in args.reference_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )[: len(watermarked_paths)]
    else:
        reference_paths = baseline_paths
    if len(reference_paths) != len(watermarked_paths):
        raise ValueError("reference image count is smaller than generated image count")

    weights = Inception_V3_Weights.DEFAULT
    inception = inception_v3(weights=weights)
    inception.fc = torch.nn.Identity()
    inception.eval().to(args.device)
    transform = weights.transforms()
    reference_features = (
        _extract_features(
            reference_paths,
            transform,
            inception,
            batch_size=args.batch_size,
            device=args.device,
        )
        if args.reference_dir
        else None
    )
    baseline_features = _extract_features(
        baseline_paths, transform, inception, batch_size=args.batch_size, device=args.device
    )
    watermarked_features = _extract_features(
        watermarked_paths, transform, inception, batch_size=args.batch_size, device=args.device
    )
    report: dict[str, Any] = {"format": "ppgs-quality-v2"}
    if args.reference_dir:
        assert reference_features is not None
        baseline_fids = _batch_fids(reference_features, baseline_features, args.batch_size)
        watermarked_fids = _batch_fids(reference_features, watermarked_features, args.batch_size)
        report["reference"] = str(args.reference_dir)
        report["fid"] = {
            "baseline": _mean_std(baseline_fids),
            "watermarked": _mean_std(watermarked_fids),
            "paired_t": paired_t_statistic(baseline_fids, watermarked_fids)
            if len(baseline_fids) >= 2
            else None,
        }
    else:
        shifts = _batch_fids(baseline_features, watermarked_features, args.batch_size)
        report["reference"] = None
        report["paired_feature_frechet_shift"] = _mean_std(shifts)
        report["fid"] = None
        report["fid_note"] = (
            "Absolute FID and its paired t-test require --reference-dir; the paper "
            "does not identify that reference corpus."
        )

    if not args.skip_clip:
        import open_clip

        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            args.clip_model,
            pretrained=args.clip_pretrained,
            device=args.device,
            cache_dir=args.clip_cache_dir,
        )
        clip_tokenizer = open_clip.get_tokenizer(args.clip_model)
        prompts = [
            line.strip()
            for line in args.prompts.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        indices = [int(path.name.split("-", 1)[0]) for path in watermarked_paths]

        def scores(paths: list[Path]) -> list[float]:
            values = []
            for batch_paths in _batched(paths, args.batch_size):
                batch_indices = [int(path.name.split("-", 1)[0]) for path in batch_paths]
                images = torch.stack(
                    [clip_preprocess(Image.open(path).convert("RGB")) for path in batch_paths]
                ).to(args.device)
                tokens = clip_tokenizer([prompts[index] for index in batch_indices]).to(
                    args.device
                )
                with torch.inference_mode():
                    image_features = clip_model.encode_image(images)
                    text_features = clip_model.encode_text(tokens)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                values.extend((image_features * text_features).sum(dim=-1).float().cpu().tolist())
            return values

        baseline_clip = scores(baseline_paths)
        watermarked_clip = scores(watermarked_paths)
        report["clip"] = {
            "model": args.clip_model,
            "pretrained": args.clip_pretrained,
            "baseline": _mean_std(baseline_clip),
            "watermarked": _mean_std(watermarked_clip),
            "paired_t": paired_t_statistic(baseline_clip, watermarked_clip),
        }

    output = args.output or args.run_dir / "quality.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
