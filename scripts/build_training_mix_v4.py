"""Build a cleaned SFT mix for NeuroCoder v4 training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        rows.append(payload)
    return rows


def _sanitize(text: str, banned_terms: list[str]) -> str:
    out = text
    for term in banned_terms:
        if not term:
            continue
        out = re.sub(re.escape(term), "Agency", out, flags=re.IGNORECASE)
    return out


def _is_degenerate(text: str) -> bool:
    clean = text.strip().lower()
    if not clean:
        return True
    if "<unk>" in clean:
        return True
    is_structured = ("<!doctype html" in clean) or ("--- a/" in clean and "+++ b/" in clean)
    if not is_structured and re.search(r"(.{1,8})\1{8,}", clean):
        return True
    words = re.findall(r"[a-z0-9_<>/#.+-]+", clean)
    if len(words) >= 80 and not is_structured:
        unique_ratio = len(set(words)) / float(len(words))
        if unique_ratio < 0.22:
            return True
    return False


def _build_anchor_rows() -> list[dict[str, str]]:
    anchors = [
        ("hi", "Hello! I am NeuroCoder. I can help with coding, patch edits, and landing page generation."),
        ("hello", "Hello! I am NeuroCoder. I can help with coding, patch edits, and landing page generation."),
        ("how are you?", "I am doing well, thank you. I am ready to help with your coding task."),
        ("Write a python function to reverse a string", "def reverse_string(value: str) -> str:\n    \"\"\"Return the reversed version of the input string.\"\"\"\n    return value[::-1]\n"),
        (
            "Provide a unified diff to change hero button color to blue-500",
            "--- a/src/components/Hero.tsx\n+++ b/src/components/Hero.tsx\n@@ -8,7 +8,7 @@ export default function Hero() {\n-        <button className=\"mt-10 rounded-lg bg-indigo-600 px-8 py-3 font-semibold hover:bg-indigo-700\">\n+        <button className=\"mt-10 rounded-lg bg-blue-500 px-8 py-3 font-semibold hover:bg-blue-600\">\n           Start Free Trial\n         </button>\n       </div>",
        ),
        (
            "Think step by step and solve 17 * 8 + 3",
            "<Think>Compute 17 * 8 first, then add 3.</Think>\n<Answer>139</Answer>",
        ),
    ]
    return [{"prompt": p, "response": r} for p, r in anchors]


def _normalize_row(
    row: dict[str, Any],
    *,
    banned_terms: list[str],
    max_prompt_chars: int,
    max_response_chars: int,
) -> dict[str, str] | None:
    prompt = str(row.get("prompt", "")).strip()
    response = str(row.get("response", "")).strip()
    if not prompt or not response:
        return None
    if len(prompt) > max_prompt_chars:
        prompt = prompt[:max_prompt_chars].strip()
    if len(response) > max_response_chars:
        response = response[:max_response_chars].strip()
    prompt = _sanitize(prompt, banned_terms)
    response = _sanitize(response, banned_terms)
    if _is_degenerate(prompt) or _is_degenerate(response):
        return None
    return {"prompt": prompt, "response": response}


def _cap_duplicates(rows: list[dict[str, str]], max_per_pair: int) -> list[dict[str, str]]:
    counts: dict[tuple[str, str], int] = {}
    out: list[dict[str, str]] = []
    for row in rows:
        key = (row["prompt"], row["response"])
        used = counts.get(key, 0)
        if used >= max_per_pair:
            continue
        counts[key] = used + 1
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cleaned NeuroCoder training mix v4")
    parser.add_argument("--sft-jsonl", default="datasets/curriculum/sft_v2.jsonl")
    parser.add_argument("--groundup-jsonl", default="datasets/groundup/neurocoder_v4.jsonl")
    parser.add_argument("--out-jsonl", default="datasets/curriculum/sft_v4_mix.jsonl")
    parser.add_argument("--manifest", default="datasets/curriculum/sft_v4_mix_manifest.json")
    parser.add_argument("--sft-max", type=int, default=24000)
    parser.add_argument("--groundup-max", type=int, default=12000)
    parser.add_argument("--max-per-pair", type=int, default=3)
    parser.add_argument("--max-prompt-chars", type=int, default=1200)
    parser.add_argument("--max-response-chars", type=int, default=14000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ban-term",
        action="append",
        default=["inferencia"],
        help="Case-insensitive term to scrub from prompts/responses.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    banned_terms = [term.strip() for term in args.ban_term if term.strip()]

    sft_rows_raw = _read_jsonl(Path(args.sft_jsonl).resolve())
    groundup_rows_raw = _read_jsonl(Path(args.groundup_jsonl).resolve())

    sft_rows: list[dict[str, str]] = []
    for row in sft_rows_raw:
        norm = _normalize_row(
            row,
            banned_terms=banned_terms,
            max_prompt_chars=args.max_prompt_chars,
            max_response_chars=args.max_response_chars,
        )
        if norm is not None:
            sft_rows.append(norm)

    groundup_rows: list[dict[str, str]] = []
    for row in groundup_rows_raw:
        norm = _normalize_row(
            row,
            banned_terms=banned_terms,
            max_prompt_chars=args.max_prompt_chars,
            max_response_chars=args.max_response_chars,
        )
        if norm is not None:
            groundup_rows.append(norm)

    random.shuffle(sft_rows)
    random.shuffle(groundup_rows)
    sft_rows = sft_rows[: args.sft_max]
    groundup_rows = groundup_rows[: args.groundup_max]

    out_rows = _build_anchor_rows() + sft_rows + groundup_rows
    out_rows = _cap_duplicates(out_rows, max_per_pair=max(1, args.max_per_pair))
    random.shuffle(out_rows)

    out_path = Path(args.out_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    manifest = {
        "out_jsonl": str(out_path),
        "rows": len(out_rows),
        "sft_source_rows": len(sft_rows_raw),
        "groundup_source_rows": len(groundup_rows_raw),
        "sft_kept": len(sft_rows),
        "groundup_kept": len(groundup_rows),
        "anchors": len(_build_anchor_rows()),
        "max_per_pair": args.max_per_pair,
        "ban_terms": banned_terms,
        "bytes": out_path.stat().st_size,
    }
    manifest_path = Path(args.manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
