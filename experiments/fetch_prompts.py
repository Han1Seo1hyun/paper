"""Fetch public prompt-only datasets through the Hugging Face dataset server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Gustavosta/Stable-Diffusion-Prompts")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--field", default="Prompt")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("experiments/prompts-1000.txt"))
    args = parser.parse_args()

    prompts: list[str] = []
    offset = args.start
    while len(prompts) < args.count:
        length = min(100, args.count - len(prompts))
        query = urlencode(
            {
                "dataset": args.dataset,
                "config": args.config,
                "split": args.split,
                "offset": offset,
                "length": length,
            }
        )
        with urlopen(
            f"https://datasets-server.huggingface.co/rows?{query}", timeout=60
        ) as response:
            payload = json.load(response)
        rows = payload.get("rows", [])
        if not rows:
            break
        for item in rows:
            value = item.get("row", {}).get(args.field)
            if isinstance(value, str) and value.strip():
                prompts.append(" ".join(value.split()))
        offset += len(rows)

    if len(prompts) < args.count:
        raise RuntimeError(f"requested {args.count} prompts, received {len(prompts)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(prompts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
