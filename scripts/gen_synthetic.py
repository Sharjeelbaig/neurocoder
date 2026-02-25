"""Generate non-LLM synthetic instruction dataset from source files."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

from data.synthetic import generate_color_edit_examples, save_examples_jsonl

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic SFT examples")
    parser.add_argument("manifest", help="Path to manifest.jsonl from ingestion")
    parser.add_argument("--out", default="datasets/synthetic/sft_color_edits.jsonl")
    parser.add_argument("--max-examples", type=int, default=100)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    files: list[tuple[str, str]] = []
    for record in records:
        source = Path(record["source_repo"]) / record["relative_path"]
        if source.exists():
            files.append((record["relative_path"], source.read_text(encoding="utf-8", errors="ignore")))

    examples = generate_color_edit_examples(files, max_examples=args.max_examples)
    if not examples:
        seed_files = [
            (
                "src/Hero.tsx",
                "export default function Hero(){return <button className='bg-blue-500 px-4 py-2 text-white'>Start</button>}",
            )
        ]
        examples = generate_color_edit_examples(seed_files, max_examples=1)

    out_path = Path(args.out)
    save_examples_jsonl(examples, out_path)
    print(f"generated {len(examples)} examples -> {out_path}")


if __name__ == "__main__":
    main()
