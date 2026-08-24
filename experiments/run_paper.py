"""Run the paper's PPGS generation, attack, inversion, and recovery loop.

This intentionally excludes FID/CLIP and external baselines; those require
large reference datasets and third-party model repositories. The produced
JSONL contains the paper's native PPGS measurements: bit accuracy, detection,
normalized inversion error, and latent Gaussian statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ppgs_watermark.attacks import apply_attack
from ppgs_watermark.diffusers_pipeline import PPGSDiffusers
from ppgs_watermark.evaluation import (
    extraction_metrics,
    gaussian_statistics,
    normalized_inversion_error,
)


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value).strip("-").lower()


def _cached_model_path(
    model_id: str, cache_dir: Path, revision: str | None = None
) -> str | Path:
    """Use a complete local snapshot directly, avoiding an unnecessary Hub probe."""

    repository_name = "models--" + model_id.replace("/", "--")
    for repository in (cache_dir / repository_name, cache_dir / "hub" / repository_name):
        if revision is not None:
            pinned = repository / "snapshots" / revision
            if (pinned / "model_index.json").is_file():
                return pinned.resolve()
        reference = repository / "refs" / "main"
        if reference.is_file():
            snapshot = repository / "snapshots" / reference.read_text(encoding="utf-8").strip()
            if (snapshot / "model_index.json").is_file():
                return snapshot.resolve()
    return model_id


def _load_pipeline(
    model_id: str,
    *,
    device: str,
    cache_dir: Path,
    variant: str | None = None,
    revision: str | None = None,
) -> Any:
    import torch
    from diffusers import StableDiffusionPipeline

    dtype = torch.float16 if device == "cuda" else torch.float32
    source = _cached_model_path(model_id, cache_dir, revision)
    load_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "cache_dir": cache_dir,
        "safety_checker": None,
        "requires_safety_checker": False,
    }
    if variant is not None:
        load_kwargs["variant"] = variant
    if revision is not None and isinstance(source, str):
        load_kwargs["revision"] = revision
    pipe = StableDiffusionPipeline.from_pretrained(source, **load_kwargs)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    elif hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if device == "cuda":
        # This is suitable for the local 6 GB GPU and requires accelerate.
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


def _write_summary(
    records: list[dict[str, Any]], attacks: list[str], output: Path
) -> None:
    summary: dict[str, dict[str, float | int]] = {}
    for attack_name in attacks:
        selected = [item for item in records if item["attack"] == attack_name]
        if not selected:
            continue
        summary[attack_name] = {
            "samples": len(selected),
            "mean_bit_accuracy": float(
                np.mean([item["bit_accuracy"] for item in selected])
            ),
            "true_positive_rate": float(
                np.mean([item["detected"] for item in selected])
            ),
            "mean_relative_l2_inversion_error": float(
                np.mean([item["normalized_inversion_error"] for item in selected])
            ),
        }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/paper.json"))
    parser.add_argument("--prompts", type=Path, default=Path("experiments/prompts.txt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--model", help="override the first model in the config")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("work/huggingface")
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        help="override configured attacks, for example: --attacks clean jpeg",
    )
    parser.add_argument("--run-name", default="main")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing records for the selected prompt range",
    )
    parser.add_argument(
        "--quality-pairs",
        action="store_true",
        help="also generate standard unwatermarked SD images for quality analysis",
    )
    parser.add_argument(
        "--artifacts",
        choices=("all", "generation", "metrics"),
        default="all",
        help="retain all attacked images, generation artifacts only, or metrics only",
    )
    parser.add_argument(
        "--payload-gamma",
        type=float,
        help="force an exact one-bit proportion for the distribution ablation",
    )
    parser.add_argument(
        "--guidance-schedule",
        choices=("exponential", "paper_exponential", "linear", "cosine"),
    )
    parser.add_argument("--maximum-guidance", type=float)
    parser.add_argument("--minimum-guidance", type=float)
    parser.add_argument("--guidance-decay", type=float)
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="rebuild summary.json from an existing metrics.jsonl without loading a model",
    )
    args = parser.parse_args()
    if args.quality_pairs and args.artifacts == "metrics":
        parser.error("--quality-pairs requires --artifacts all or generation")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.guidance_schedule is not None:
        config["guidance_schedule"] = args.guidance_schedule
    if args.maximum_guidance is not None:
        config["maximum_guidance"] = args.maximum_guidance
    if args.minimum_guidance is not None:
        config["minimum_guidance"] = args.minimum_guidance
    if args.guidance_decay is not None:
        config["guidance_decay"] = args.guidance_decay
    model_id = args.model or config["models"][0]
    model_source = config.get("model_sources", {}).get(model_id, model_id)
    model_variant = config.get("model_variants", {}).get(model_id)
    model_revision = config.get("model_revisions", {}).get(model_id)
    attacks = args.attacks or config["attacks"]
    prompts = [
        line.strip() for line in args.prompts.read_text(encoding="utf-8").splitlines()
    ]
    prompts = [value for value in prompts if value and not value.startswith("#")]
    indexed_prompts = list(enumerate(prompts))[args.start : args.start + args.limit]
    if not indexed_prompts:
        raise ValueError("no prompts were selected")

    output = args.output / _slug(args.run_name) / _slug(model_id)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "metrics.jsonl"

    import accelerate
    import diffusers
    import torch
    import transformers

    run_metadata = {
        "format": "ppgs-paper-run-v1",
        "model": model_id,
        "model_source": model_source,
        "model_variant": model_variant,
        "model_revision": model_revision,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": {
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "numpy": np.__version__,
        },
        "config": config,
        "config_file_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "selected_attacks": attacks,
        "prompt_count": len(indexed_prompts),
        "prompt_file": str(args.prompts),
        "prompt_file_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "prompt_start": args.start,
        "quality_pairs": args.quality_pairs,
        "artifacts": args.artifacts,
        "payload_gamma": args.payload_gamma,
    }
    (output / "run.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.summarize_only:
        records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        _write_summary(records, sorted({item["attack"] for item in records}), output)
        return

    pipeline = _load_pipeline(
        model_source,
        device=args.device,
        cache_dir=args.cache_dir,
        variant=model_variant,
        revision=model_revision,
    )
    ppgs = PPGSDiffusers(pipeline)

    records: list[dict[str, Any]] = []
    if args.resume and records_path.exists():
        records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if args.force:
        selected_indices = {index for index, _ in indexed_prompts}
        records = [
            item for item in records if item["prompt_index"] not in selected_indices
        ]
        records_path.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
            encoding="utf-8",
        )
    completed = {(item["prompt_index"], item["attack"]) for item in records}

    for prompt_index, prompt in indexed_prompts:
        if all((prompt_index, attack_name) in completed for attack_name in attacks):
            continue
        payload_rng = np.random.default_rng(10_000 + prompt_index)
        payload_length = int(config["payload_length"])
        if args.payload_gamma is None:
            watermark = payload_rng.integers(0, 2, payload_length, dtype=np.uint8)
        else:
            if not 0 <= args.payload_gamma <= 1:
                raise ValueError("payload_gamma must be in [0, 1]")
            one_count = round(payload_length * args.payload_gamma)
            watermark = np.concatenate(
                [
                    np.ones(one_count, dtype=np.uint8),
                    np.zeros(payload_length - one_count, dtype=np.uint8),
                ]
            )
            payload_rng.shuffle(watermark)
        generated = ppgs.generate(
            prompt,
            watermark,
            height=int(config["height"]),
            width=int(config["width"]),
            num_inference_steps=int(config["generation_steps"]),
            maximum_guidance=float(config["maximum_guidance"]),
            minimum_guidance=float(config["minimum_guidance"]),
            guidance_decay=float(config["guidance_decay"]),
            guidance_schedule=config["guidance_schedule"],
            public_seed=int(config["public_seed"]),
            sampling_seed=20_000 + prompt_index,
            bits_per_position=int(config["bits_per_position"]),
            payload_layout=config["payload_layout"],
            spatial_copies=tuple(config["spatial_copies"]),
        )
        stem = f"{prompt_index:04d}"
        reference = generated.terminal_latents.detach().float().cpu().numpy()
        if args.artifacts != "metrics":
            generated.images[0].save(output / f"{stem}-watermarked.png")
            generated.save_manifest(output / f"{stem}-manifest.json")
            np.save(output / f"{stem}-terminal-latent.npy", reference)

        if args.quality_pairs:
            import torch

            clean_generator = torch.Generator(device="cpu").manual_seed(
                40_000 + prompt_index
            )
            clean_result = pipeline(
                prompt,
                height=int(config["height"]),
                width=int(config["width"]),
                num_inference_steps=int(config["generation_steps"]),
                guidance_scale=float(config["maximum_guidance"]),
                generator=clean_generator,
            )
            clean_result.images[0].save(output / f"{stem}-baseline.png")

        for attack_index, attack_name in enumerate(attacks):
            if (prompt_index, attack_name) in completed:
                continue
            attack_seed = 30_000 + prompt_index * 100 + attack_index
            attacked = apply_attack(generated.images[0], attack_name, seed=attack_seed)
            if args.artifacts == "all":
                attacked.save(output / f"{stem}-{attack_name}.png")
            inverted = ppgs.invert(
                attacked,
                num_inference_steps=int(config["inversion_steps"]),
                metadata=generated.metadata,
            )
            recovered = ppgs.extract(inverted, generated.metadata)
            inverted_np = inverted.detach().float().cpu().numpy()
            record: dict[str, Any] = {
                "model": model_id,
                "prompt_index": prompt_index,
                "prompt": prompt,
                "attack": attack_name,
                "expected_payload": "".join(str(int(bit)) for bit in watermark),
                "recovered_payload": "".join(str(int(bit)) for bit in recovered),
                **extraction_metrics(
                    watermark,
                    recovered,
                    false_positive_rate=float(config["target_false_positive_rate"]),
                ),
                "normalized_inversion_error": normalized_inversion_error(
                    reference, inverted_np
                ),
                "terminal_latent": gaussian_statistics(reference),
            }
            records.append(record)
            records_path.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
                encoding="utf-8",
            )

    _write_summary(records, list(attacks), output)


if __name__ == "__main__":
    main()
