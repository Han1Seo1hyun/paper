"""Aggregate PPGS JSONL runs into tables, detection curves, and attribution."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ppgs_watermark.analysis import detection_curve, user_scale_attribution


def _bits(value: str) -> list[int]:
    return [int(bit) for bit in value]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--output", type=Path, default=Path("outputs/paper/report"))
    parser.add_argument("--target-fpr", type=float, default=1e-6)
    parser.add_argument(
        "--paper-targets", type=Path, default=Path("configs/paper_table1.json")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    table_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    attribution: dict[str, Any] = {}
    for metrics_path in sorted(args.root.glob("*/*/metrics.jsonl")):
        run_name = metrics_path.parent.parent.name
        model_slug = metrics_path.parent.name
        records = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records:
            continue
        for attack in sorted({record["attack"] for record in records}):
            selected = [record for record in records if record["attack"] == attack]
            scores = [float(record["bit_accuracy"]) for record in selected]
            curve = detection_curve(scores, 256)
            operating = min(
                (point for point in curve if point["false_positive_rate"] <= args.target_fpr),
                key=lambda point: abs(point["false_positive_rate"] - args.target_fpr),
            )
            table_rows.append(
                {
                    "run": run_name,
                    "model": model_slug,
                    "attack": attack,
                    "samples": len(selected),
                    "bit_accuracy_mean": float(np.mean(scores)),
                    "bit_accuracy_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                    "tpr_at_target_fpr": float(
                        np.mean([score >= operating["threshold"] for score in scores])
                    ),
                    "threshold": operating["threshold"],
                    "theoretical_fpr": operating["false_positive_rate"],
                    "inversion_error_mean": float(
                        np.mean([record["normalized_inversion_error"] for record in selected])
                    ),
                }
            )
            for point in curve:
                curve_rows.append(
                    {"run": run_name, "model": model_slug, "attack": attack, **point}
                )

        clean = [
            record
            for record in records
            if record["attack"] == "clean"
            and "expected_payload" in record
            and "recovered_payload" in record
        ]
        if clean:
            first = clean[0]
            attribution[f"{run_name}/{model_slug}"] = user_scale_attribution(
                _bits(first["recovered_payload"]),
                _bits(first["expected_payload"]),
                [10, 100, 1_000, 10_000, 100_000, 1_000_000],
                seed=2026,
            )

    if not table_rows:
        raise ValueError(f"no metrics.jsonl files found below {args.root}")
    _write_csv(args.output / "attack-table.csv", table_rows)
    _write_csv(args.output / "detection-curves.csv", curve_rows)
    report = {
        "format": "ppgs-aggregate-v1",
        "target_fpr": args.target_fpr,
        "paper_targets": json.loads(args.paper_targets.read_text(encoding="utf-8"))
        if args.paper_targets.is_file()
        else None,
        "runs": table_rows,
        "user_scale_attribution": attribution,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
