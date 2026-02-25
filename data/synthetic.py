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
_SPACING_RE = re.compile(r"\b(p|px|py|m|mx|my)-(\d{1,2})\b")

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
        patch = ""
        instruction = ""
        metadata: dict[str, str] = {"generator": "rule_template_v1"}

        color_match = _COLOR_RE.search(content)
        spacing_match = _SPACING_RE.search(content)

        if color_match:
            prefix = color_match.group(1)
            old_color = color_match.group(0)
            new_color = f"{prefix}-{_NEW_COLORS[idx % len(_NEW_COLORS)]}"
            updated = content.replace(old_color, new_color, 1)
            patch = generate_unified_diff(path, content, updated)
            instruction = (
                f"Change {prefix} color in {Path(path).name} from {old_color} to {new_color}. "
                "Return a unified diff patch only."
            )
            metadata["old_color"] = old_color
            metadata["new_color"] = new_color
        elif spacing_match:
            old_spacing = spacing_match.group(0)
            axis = spacing_match.group(1)
            new_spacing = f"{axis}-{(int(spacing_match.group(2)) + 2) % 10 or 4}"
            updated = content.replace(old_spacing, new_spacing, 1)
            patch = generate_unified_diff(path, content, updated)
            instruction = (
                f"Adjust spacing in {Path(path).name} from {old_spacing} to {new_spacing}. "
                "Return a unified diff patch only."
            )
            metadata["old_spacing"] = old_spacing
            metadata["new_spacing"] = new_spacing
        else:
            continue

        file_map = {path: content}
        apply_result, lint_ok, build_ok, notes = apply_and_validate(file_map, patch)
        if not (apply_result.ok and lint_ok and build_ok):
            continue

        example = TrainExample(
            id=f"syn-edit-{idx}",
            source_license="SYNTHETIC",
            task_type="patch_edit",
            instruction=instruction,
            context_files=[SourceFile(path=path, content=content)],
            target_patch=patch,
            metadata={**metadata, "quality_filter_notes": notes},
        )
        examples.append(example)

    return examples


def save_examples_jsonl(examples: list[TrainExample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(asdict(example), sort_keys=True) + "\n")
