"""Run keyed Gaussian Shading and single-bit Tree-Ring baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ppgs_watermark.attacks import apply_attack
from ppgs_watermark.baselines import (
    gaussian_shading_decode,
    gaussian_shading_embed,
    tree_ring_distance,
    tree_ring_embed,
    tree_ring_p_value,
)
from ppgs_watermark.evaluation import extraction_metrics
from ppgs_watermark.diffusers_pipeline import PPGSDiffusers

from run_paper import _load_pipeline, _slug


def _generate(pipe: Any, prompt: str, latent: np.ndarray, steps: int, guidance: float) -> Any:
    import torch

    dtype = next(pipe.unet.parameters()).dtype
    tensor = torch.as_tensor(latent, device=pipe._execution_device, dtype=dtype)
    return pipe(
        prompt,
        latents=tensor,
        height=512,
        width=512,
        num_inference_steps=steps,
        guidance_scale=guidance,
    ).images[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/paper.json"))
    parser.add_argument("--prompts", type=Path, default=Path("experiments/prompts-1000.txt"))
    parser.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--methods", nargs="+", choices=("gaussian-shading", "tree-ring"), default=["gaussian-shading", "tree-ring"])
    parser.add_argument("--attacks", nargs="+", default=["clean", "composite"])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--cache-dir", type=Path, default=Path("work/huggingface"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    prompts = [line.strip() for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = list(enumerate(prompts))[args.start : args.start + args.limit]
    model_source = config.get("model_sources", {}).get(args.model, args.model)
    model_variant = config.get("model_variants", {}).get(args.model)
    model_revision = config.get("model_revisions", {}).get(args.model)
    pipe = _load_pipeline(
        model_source,
        device=args.device,
        cache_dir=args.cache_dir,
        variant=model_variant,
        revision=model_revision,
    )
    inversion = PPGSDiffusers(pipe)

    for method in args.methods:
        output = args.output / method / _slug(args.model)
        output.mkdir(parents=True, exist_ok=True)
        (output / "run.json").write_text(
            json.dumps(
                {
                    "format": "ppgs-latent-baseline-run-v1",
                    "method": method,
                    "model": args.model,
                    "model_source": model_source,
                    "model_variant": model_variant,
                    "model_revision": model_revision,
                    "prompt_start": args.start,
                    "prompt_count": len(selected),
                    "attacks": args.attacks,
                    "generation_steps": int(config["generation_steps"]),
                    "inversion_steps": int(config["inversion_steps"]),
                    "guidance_scale": float(config["maximum_guidance"]),
                    "watermark_seed": 999999,
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
            records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        completed = {(item["prompt_index"], item["attack"]) for item in records}

        for prompt_index, prompt in selected:
            if all((prompt_index, attack) in completed for attack in args.attacks):
                continue
            payload = np.random.default_rng(90_000 + prompt_index).integers(0, 2, 256, dtype=np.uint8)
            if method == "gaussian-shading":
                latent, secret = gaussian_shading_embed(
                    payload, key_seed=999_999, sampling_seed=100_000 + prompt_index
                )
            else:
                native = np.random.default_rng(100_000 + prompt_index).standard_normal((1, 4, 64, 64))
                latent, secret = tree_ring_embed(native, seed=999_999)
            image = _generate(
                pipe,
                prompt,
                latent,
                int(config["generation_steps"]),
                float(config["maximum_guidance"]),
            )
            image.save(output / f"{prompt_index:04d}-watermarked.png")

            for attack_index, attack in enumerate(args.attacks):
                if (prompt_index, attack) in completed:
                    continue
                attacked = apply_attack(image, attack, seed=110_000 + prompt_index * 100 + attack_index)
                attacked.save(output / f"{prompt_index:04d}-{attack}.png")
                reversed_latent = inversion.invert(
                    attacked, num_inference_steps=int(config["inversion_steps"])
                ).detach().float().cpu().numpy()
                if method == "gaussian-shading":
                    recovered = gaussian_shading_decode(reversed_latent, secret)
                    metrics: dict[str, Any] = extraction_metrics(
                        payload, recovered, false_positive_rate=float(config["target_false_positive_rate"])
                    )
                else:
                    p_value = tree_ring_p_value(reversed_latent, secret)
                    metrics = {
                        "tree_ring_l1_distance": tree_ring_distance(reversed_latent, secret),
                        "tree_ring_p_value": p_value,
                        "detected": p_value <= float(config["target_false_positive_rate"]),
                        "target_false_positive_rate": float(config["target_false_positive_rate"]),
                    }
                records.append({"method": method, "model": args.model, "prompt_index": prompt_index, "attack": attack, **metrics})
                records_path.write_text(
                    "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
                    encoding="utf-8",
                )

        summary = {}
        for attack in args.attacks:
            subset = [item for item in records if item["attack"] == attack]
            if not subset:
                continue
            if method == "gaussian-shading":
                summary[attack] = {
                    "samples": len(subset),
                    "mean_bit_accuracy": float(np.mean([item["bit_accuracy"] for item in subset])),
                    "true_positive_rate": float(np.mean([item["detected"] for item in subset])),
                }
            else:
                summary[attack] = {
                    "samples": len(subset),
                    "mean_tree_ring_l1_distance": float(np.mean([item["tree_ring_l1_distance"] for item in subset])),
                    "true_positive_rate": float(np.mean([item["detected"] for item in subset])),
                }
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
