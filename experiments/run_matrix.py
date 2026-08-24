"""Run the paper model matrix and ablations as resumable subprocess jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/paper.json"))
    parser.add_argument(
        "--prompts", type=Path, default=Path("experiments/prompts-1000.txt")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("work/huggingface")
    )
    parser.add_argument("--quality-pairs", action="store_true")
    parser.add_argument(
        "--artifacts", choices=("all", "generation", "metrics"), default="all"
    )
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config: dict[str, Any] = json.loads(args.config.read_text(encoding="utf-8"))
    runner = Path(__file__).with_name("run_paper.py")
    common = [
        sys.executable,
        str(runner),
        "--config",
        str(args.config),
        "--prompts",
        str(args.prompts),
        "--output",
        str(args.output),
        "--limit",
        str(args.limit),
        "--start",
        str(args.start),
        "--device",
        args.device,
        "--cache-dir",
        str(args.cache_dir),
        "--artifacts",
        args.artifacts,
        "--resume",
    ]
    if args.quality_pairs:
        common.append("--quality-pairs")

    for model in config["models"]:
        _run(common + ["--run-name", "main", "--model", model], dry_run=args.dry_run)

    if args.include_ablations:
        for model in config["models"]:
            for schedule in ("exponential", "linear", "cosine", "paper_exponential"):
                _run(
                    common
                    + [
                        "--run-name",
                        f"ablation-guidance-{schedule}",
                        "--model",
                        model,
                        "--guidance-schedule",
                        schedule,
                    ],
                    dry_run=args.dry_run,
                )
            _run(
                common
                + [
                    "--run-name",
                    "ablation-guidance-none",
                    "--model",
                    model,
                    "--maximum-guidance",
                    "1.0",
                    "--minimum-guidance",
                    "1.0",
                ],
                dry_run=args.dry_run,
            )
            for gamma in (0.1, 0.3, 0.5, 0.7, 0.9):
                _run(
                    common
                    + [
                        "--run-name",
                        f"ablation-gamma-{gamma:.2f}",
                        "--model",
                        model,
                        "--payload-gamma",
                        str(gamma),
                    ],
                    dry_run=args.dry_run,
                )


if __name__ == "__main__":
    main()
