"""Create a human-readable, evidence-backed reproduction report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _value(summary: dict[str, Any], attack: str, key: str) -> str:
    value = summary.get(attack, {}).get(key)
    return "not run" if value is None else f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--baselines", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--targets", type=Path, default=Path("configs/paper_table1.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/paper/REPRODUCTION_REPORT.md"))
    args = parser.parse_args()

    targets = _json(args.targets)
    target = targets["ppgs"]
    rows = []
    for directory in sorted((args.root / "main").glob("*")):
        summary_path = directory / "summary.json"
        run_path = directory / "run.json"
        if not summary_path.is_file() or not run_path.is_file():
            continue
        summary = _json(summary_path)
        run = _json(run_path)
        model_index = {
            "CompVis/stable-diffusion-v1-4": 0,
            "stabilityai/stable-diffusion-2-base": 1,
            "stabilityai/stable-diffusion-2-1-base": 2,
        }.get(run["model"], 0)
        rows.append(
            "| {model} | {samples} | {clean_tpr} | {paper_clean} | {adv_tpr} | {paper_adv} | {clean_acc} | {adv_acc} |".format(
                model=run["model"],
                samples=summary.get("clean", {}).get("samples", 0),
                clean_tpr=_value(summary, "clean", "true_positive_rate"),
                paper_clean=f"{target['tpr_clean'][model_index]:.3f}",
                adv_tpr=_value(summary, "composite", "true_positive_rate"),
                paper_adv=f"{target['tpr_adversarial'][model_index]:.3f}",
                clean_acc=_value(summary, "clean", "mean_bit_accuracy"),
                adv_acc=_value(summary, "composite", "mean_bit_accuracy"),
            )
        )

    baseline_rows = []
    for summary_path in sorted(args.baselines.glob("**/summary.json")):
        summary = _json(summary_path)
        name = summary_path.parent.relative_to(args.baselines).as_posix()
        baseline_rows.append(
            f"| {name} | {_value(summary, 'clean', 'true_positive_rate')} | "
            f"{_value(summary, 'composite', 'true_positive_rate')} | "
            f"{_value(summary, 'clean', 'mean_bit_accuracy')} | "
            f"{_value(summary, 'composite', 'mean_bit_accuracy')} |"
        )

    quality_rows = []
    for quality_path in sorted((args.root / "main").glob("*/quality.json")):
        quality = _json(quality_path)
        clip = quality.get("clip", {})
        shift = quality.get("paired_feature_frechet_shift", {})
        quality_rows.append(
            f"| {quality_path.parent.name} | {clip.get('baseline', {}).get('mean', float('nan')):.6f} | "
            f"{clip.get('watermarked', {}).get('mean', float('nan')):.6f} | "
            f"{clip.get('paired_t', float('nan')):.4f} | {shift.get('mean', float('nan')):.4f} |"
        )
    distribution_path = args.root / "distribution.json"
    distribution_rows = []
    if distribution_path.is_file():
        for gamma, values in _json(distribution_path)["gamma"].items():
            distribution_rows.append(
                f"| {gamma} | {values['realized_gamma']:.6f} | {values['abs_mean']:.6f} | "
                f"{values['abs_std_minus_one']:.6f} | {values['ks_statistic']:.6f} | "
                f"{values['ks_pvalue_approx']:.4f} |"
            )

    lines = [
        "# PPGS independent reproduction report",
        "",
        "This report separates measurements produced by this repository from the numbers printed in the paper.",
        "",
        "## PPGS model matrix",
        "",
        "| Model | N | clean TPR | paper | composite TPR | paper | clean bit acc. | composite bit acc. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *(rows or ["| not run | 0 | – | – | – | – | – | – |"]),
        "",
        "## Public baseline runs",
        "",
        "| Method/model | clean TPR | composite TPR | clean bit acc. | composite bit acc. |",
        "|---|---:|---:|---:|---:|",
        *(baseline_rows or ["| not run | – | – | – | – |"]),
        "",
        "## Quality output",
        "",
        "| Model | baseline CLIP | PPGS CLIP | paired t | paired feature Fréchet shift |",
        "|---|---:|---:|---:|---:|",
        *(quality_rows or ["| not run | – | – | – | – |"]),
        "",
        "Absolute FID is intentionally not reported because the paper does not identify its real-image reference corpus. CLIP uses the declared ViT-B-32/laion2b_s34b_b79k checkpoint.",
        "",
        "## Gaussian distribution ablation",
        "",
        "| requested γ | realized γ | |mean| | |std-1| | KS D | KS p |",
        "|---:|---:|---:|---:|---:|---:|",
        *(distribution_rows or ["| not run | – | – | – | – | – |"]),
        "",
        "## Reproducibility boundary",
        "",
        "- The paper does not identify the real-image FID reference corpus, batch size, Inception preprocessing/version, or CLIP checkpoint.",
        "- The composite attack order and per-stage sampled strengths are not specified.",
        "- Table 5 names DiffusionDB and LAION-Aesthetics but not their repository IDs, subsets, prompt counts, row selections, or splits; a deterministic independent selection must therefore be declared by the reproducer.",
        "- The stated 256-bit payload conflicts with Algorithms 1/3, which assign one bit to each of 16,384 latent positions. This implementation follows Gaussian Shading's 4x8x8 payload tiled over 4x64x64.",
        "- The paper states both 1e-10 and 1e-6 FPR; this report follows Table 1 at 1e-6.",
        "- The paper names a normalized inversion error but does not define its normalization. This implementation reports ||z_inv-z_T||_2 / ||z_T||_2 and therefore does not compare that number directly to Table 2.",
        "- RivaGAN's public checkpoint carries 32 bits and Tree-Ring is a single tag; these are not silently reported as 256-bit methods.",
        "- HiDDeN, MBRS, and Stable Signature require method-specific pretrained checkpoints or fine-tuned decoders that the paper does not identify. Their published numbers are targets, not results generated here, unless corresponding artifacts are supplied.",
        "- The original stabilityai SD 2.0/2.1 repositories return HTTP 401 without a licensed Hugging Face session in this environment. Runs therefore record the public sd-research fp16 mirror as `model_source`; supply authenticated originals to remove this provenance difference.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
