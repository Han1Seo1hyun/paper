"""Audit a run for missing/duplicate records and degenerate saved images."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageStat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-prompts", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    attacks = run["selected_attacks"]
    records = [
        json.loads(line)
        for line in (args.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keys = [(item["prompt_index"], item["attack"]) for item in records]
    counts = Counter(keys)
    indices = sorted({item["prompt_index"] for item in records})
    missing = [
        [index, attack]
        for index in range(run["prompt_start"], run["prompt_start"] + args.expected_prompts)
        for attack in attacks
        if counts[(index, attack)] == 0
    ]
    duplicates = [[index, attack, count] for (index, attack), count in counts.items() if count > 1]
    degenerate = []
    for path in sorted(args.run_dir.glob("*.png")):
        statistics = ImageStat.Stat(Image.open(path).convert("RGB"))
        if max(statistics.rms) < 2.0:
            degenerate.append(path.name)
    report = {
        "format": "ppgs-run-audit-v1",
        "records": len(records),
        "prompt_indices": indices,
        "missing": missing,
        "duplicates": duplicates,
        "degenerate_images": degenerate,
        "passed": not missing and not duplicates and not degenerate,
    }
    output = args.output or args.run_dir / "audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
