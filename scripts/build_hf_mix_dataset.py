"""Build a sliced alignment dataset from selected Hugging Face datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

# Avoid importing local ./datasets folder instead of huggingface datasets package.
REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in ("", str(REPO_ROOT)):
    if candidate in sys.path:
        sys.path.remove(candidate)

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset


DEFAULT_SOURCES = [
    ("smirki/UIGEN-T1.1-TAILWIND", "tailwind"),
    ("crownelius/GLM-5.0-25000x", "reasoning"),
    ("crownelius/Opus-4.6-Reasoning-3300x", "reasoning"),
]

ANCHOR_BLOCKS = [
    "User: hi\nAssistant: Hello! I am NeuroCoder. I can help with coding, patch edits, and landing page generation.",
    "User: how are you?\nAssistant: I am doing well, thank you. I am ready to help with your coding task.",
    (
        "User: think step by step and solve 17 * 8 + 3\n"
        "Assistant: <thinking>Compute 17 * 8 first, then add 3.</thinking>\n<answer>139</answer>"
    ),
    (
        "User: change primary color in Hero button and return unified diff\n"
        "Assistant: --- a/src/components/Hero.tsx\n"
        "+++ b/src/components/Hero.tsx\n"
        "@@ -8,7 +8,7 @@ export default function Hero() {\n"
        "-        <button className=\"mt-10 rounded-lg bg-indigo-600 px-8 py-3 font-semibold hover:bg-indigo-700\">\n"
        "+        <button className=\"mt-10 rounded-lg bg-emerald-500 px-8 py-3 font-semibold hover:bg-emerald-600\">\n"
        "           Start Free Trial\n"
        "         </button>\n"
        "       </div>"
    ),
]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = []
        for item in value:
            text = _as_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        # Prefer "content"/"text"/"value" fields when present.
        for key in ("content", "text", "value", "output", "response", "answer"):
            if key in value:
                text = _as_text(value.get(key))
                if text:
                    return text
        return ""
    return ""


def _conversation_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    for key in ("messages", "conversation", "conversations", "chat"):
        payload = row.get(key)
        if not isinstance(payload, list):
            continue
        user_text = ""
        assistant_text = ""
        for item in payload:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", item.get("from", item.get("speaker", "")))).lower()
            text = _as_text(item)
            if not text:
                continue
            if role in {"user", "human", "prompt"}:
                user_text = text
            elif role in {"assistant", "model", "gpt"} and user_text:
                assistant_text = text
                break
        if user_text and assistant_text:
            return user_text, assistant_text
    return None


def _extract_pair(row: dict[str, Any], source_kind: str) -> tuple[str, str] | None:
    convo = _conversation_pair(row)
    if convo:
        return convo

    prompt_keys = ("instruction", "prompt", "question", "query", "input", "user", "problem")
    response_keys = (
        "response",
        "output",
        "answer",
        "completion",
        "assistant",
        "target",
        "code",
        "solution",
    )

    prompt = ""
    response = ""
    for key in prompt_keys:
        prompt = _as_text(row.get(key))
        if prompt:
            break
    for key in response_keys:
        response = _as_text(row.get(key))
        if response:
            break

    if source_kind == "reasoning":
        thinking = _as_text(row.get("thinking"))
        solution = _as_text(row.get("solution"))
        if (thinking or solution) and not response:
            response = solution or thinking
        if prompt and (thinking or solution):
            think_text = thinking or "Work through the problem carefully."
            answer_text = solution or response
            response = f"<thinking>{think_text}</thinking>\n<answer>{answer_text}</answer>"

    if not prompt and source_kind == "tailwind":
        prompt = "Generate a React + Tailwind landing page."
    if not response:
        text = _as_text(row.get("text"))
        if text:
            if source_kind == "reasoning":
                prompt = prompt or "Solve this task with concise reasoning and final answer."
                response = text
            elif ("<html" in text.lower()) or ("class=" in text and "tailwind" in text.lower()):
                prompt = prompt or "Generate a React + Tailwind landing page."
                response = text

    prompt = prompt.strip()
    response = response.strip()
    if not prompt or not response:
        return None

    if source_kind == "reasoning" and "<thinking>" not in response.lower():
        compact = re.sub(r"\s+", " ", response).strip()
        response = f"<thinking>Work through the problem carefully.</thinking>\n<answer>{compact}</answer>"

    return prompt, response


def _resolve_stream(dataset_id: str, split: str | None, config: str | None):
    resolved_config = config
    resolved_split = split

    if resolved_config is None:
        try:
            configs = get_dataset_config_names(dataset_id)
            if configs:
                resolved_config = configs[0]
        except Exception:
            resolved_config = None

    if resolved_split is None:
        try:
            split_names = get_dataset_split_names(dataset_id, config_name=resolved_config)
            if split_names:
                resolved_split = "train" if "train" in split_names else split_names[0]
        except Exception:
            resolved_split = "train"

    kwargs: dict[str, Any] = {"split": resolved_split or "train", "streaming": True}
    if resolved_config:
        kwargs["name"] = resolved_config
    return load_dataset(dataset_id, **kwargs)


def build_blocks(
    *,
    dataset_id: str,
    source_kind: str,
    per_dataset: int,
    max_chars: int,
    split: str | None,
    config: str | None,
) -> tuple[list[str], int]:
    stream = _resolve_stream(dataset_id, split=split, config=config)
    blocks: list[str] = []
    scanned = 0
    for row in stream:
        scanned += 1
        if not isinstance(row, dict):
            continue
        pair = _extract_pair(row, source_kind=source_kind)
        if pair is None:
            continue
        prompt, response = pair
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars].strip()
        if len(response) > max_chars:
            response = response[:max_chars].strip()
        if len(prompt) < 2 or len(response) < 2:
            continue
        block = f"User: {prompt}\nAssistant: {response}"
        blocks.append(block)
        if len(blocks) >= per_dataset:
            break
    return blocks, scanned


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sliced HF mix dataset for NeuroCoder")
    parser.add_argument("--out", default="datasets/curriculum/hf_mix_v1.txt")
    parser.add_argument("--manifest", default="datasets/curriculum/hf_mix_v1_manifest.json")
    parser.add_argument("--per-dataset", type=int, default=900)
    parser.add_argument("--max-chars", type=int, default=2200)
    parser.add_argument("--split", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    all_blocks: list[str] = []
    manifest: dict[str, Any] = {
        "out": str(Path(args.out).resolve()),
        "per_dataset": args.per_dataset,
        "sources": [],
    }

    for dataset_id, source_kind in DEFAULT_SOURCES:
        blocks, scanned = build_blocks(
            dataset_id=dataset_id,
            source_kind=source_kind,
            per_dataset=args.per_dataset,
            max_chars=args.max_chars,
            split=args.split,
            config=None,
        )
        all_blocks.extend(blocks)
        manifest["sources"].append(
            {
                "dataset": dataset_id,
                "kind": source_kind,
                "kept": len(blocks),
                "scanned": scanned,
            }
        )

    # Keep anchor task behavior stable while adding external data.
    all_blocks.extend(ANCHOR_BLOCKS * 60)
    random.shuffle(all_blocks)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(all_blocks) + "\n", encoding="utf-8")

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["blocks"] = len(all_blocks)
    manifest["bytes"] = out_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
