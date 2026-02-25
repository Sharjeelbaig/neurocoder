"""Non-LLM synthetic instruction generation for SFT and preference datasets."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Iterable

from infer.diff_utils import generate_unified_diff
from infer.schemas import SourceFile, TrainExample
from infer.validators import apply_and_validate

_COLOR_RE = re.compile(
    r"\b(bg|text|border)-(slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-\d{2,3}\b"
)

_NEW_COLORS = [
    "emerald-500",
    "sky-500",
    "violet-500",
    "rose-500",
    "amber-500",
    "teal-500",
]


def generate_color_edit_examples(
    files: Iterable[tuple[str, str]],
    max_examples: int = 100,
) -> list[TrainExample]:
    examples: list[TrainExample] = []

    for idx, (path, content) in enumerate(files):
        if idx >= max_examples:
            break
        match = _COLOR_RE.search(content)
        if not match:
            continue

        prefix = match.group(1)
        old_color = match.group(0)
        new_color = f"{prefix}-{_NEW_COLORS[idx % len(_NEW_COLORS)]}"
        updated = content.replace(old_color, new_color, 1)
        patch = generate_unified_diff(path, content, updated)

        file_map = {path: content}
        apply_result, lint_ok, build_ok, notes = apply_and_validate(file_map, patch)
        if not (apply_result.ok and lint_ok and build_ok):
            continue

        instruction = (
            f"Change {prefix} color in {Path(path).name} from {old_color} to {new_color}. "
            "Return a unified diff patch only."
        )

        example = TrainExample(
            id=f"syn-color-{idx}",
            source_license="SYNTHETIC",
            task_type="patch_edit",
            instruction=instruction,
            context_files=[SourceFile(path=path, content=content)],
            target_patch=patch,
            metadata={
                "generator": "rule_template_v1",
                "quality_filter_notes": notes,
                "old_color": old_color,
                "new_color": new_color,
            },
        )
        examples.append(example)

    return examples


def save_examples_jsonl(examples: list[TrainExample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(asdict(example), sort_keys=True) + "\n")
