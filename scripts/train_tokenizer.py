"""Train and evaluate from-scratch tokenizer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

from train.tokenizer import evaluate_tokenizer_quality, train_simple_tokenizer

def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyMoE tokenizer")
    parser.add_argument("corpus", nargs="+", help="Text/code file paths")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--out", default="artifacts/tokenizer/tokenizer.json")
    args = parser.parse_args()

    corpus_files: list[Path] = []
    for entry in args.corpus:
        path = Path(entry)
        if path.name == "manifest.jsonl":
            for raw in path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                record = json.loads(raw)
                source = Path(record["source_repo"]) / record["relative_path"]
                if source.exists():
                    corpus_files.append(source)
            continue
        corpus_files.append(path)

    if not corpus_files:
        raise SystemExit("no corpus files found")

    tokenizer = train_simple_tokenizer(corpus_files=corpus_files, vocab_size=args.vocab_size)
    out_path = Path(args.out)
    tokenizer.to_json(out_path)

    samples = [path.read_text(encoding="utf-8", errors="ignore")[:2000] for path in corpus_files[:20]]
    quality = evaluate_tokenizer_quality(tokenizer, samples)
    print(f"saved tokenizer to {out_path}")
    print(f"coverage={quality.coverage:.4f} fragmentation={quality.fragmentation:.4f} reserved_ok={quality.reserved_ok}")


if __name__ == "__main__":
    main()
