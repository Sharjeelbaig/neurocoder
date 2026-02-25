"""Deterministic validators for patch application and lightweight code health checks."""

from __future__ import annotations

import re
from typing import Iterable

from infer.diff_utils import ApplyResult, apply_unified_diff


_TAILWIND_COLOR_RE = re.compile(r"\b(bg|text|border)-(slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-\d{2,3}\b")


def _balanced_symbols(text: str) -> bool:
    pairs = {")": "(", "}": "{", "]": "["}
    stack: list[str] = []
    for char in text:
        if char in "({[":
            stack.append(char)
            continue
        if char in pairs:
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
    return not stack


def lint_react_tailwind(files: dict[str, str]) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True
    for path, content in files.items():
        if not path.endswith((".tsx", ".jsx", ".ts", ".js", ".css")):
            continue
        if "\t" in content:
            ok = False
            notes.append(f"{path}: tab characters are disallowed")
        if "className" in content and "tailwind" not in content.lower():
            # soft note only; tailwind marker may not be present.
            notes.append(f"{path}: className present; ensure Tailwind setup exists")
        if "className" in content and not _TAILWIND_COLOR_RE.search(content):
            notes.append(f"{path}: no Tailwind color token detected")
    return ok, notes


def build_check(files: dict[str, str]) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True
    for path, content in files.items():
        if path.endswith((".tsx", ".jsx", ".ts", ".js")):
            if not _balanced_symbols(content):
                ok = False
                notes.append(f"{path}: unbalanced symbols")
            if "BROKEN_BUILD" in content:
                ok = False
                notes.append(f"{path}: BROKEN_BUILD marker found")
    return ok, notes


def apply_and_validate(
    files: dict[str, str],
    patch: str,
) -> tuple[ApplyResult, bool, bool, list[str]]:
    apply_result = apply_unified_diff(files, patch)
    if not apply_result.ok:
        return apply_result, False, False, apply_result.notes

    lint_ok, lint_notes = lint_react_tailwind(apply_result.files)
    build_ok, build_notes = build_check(apply_result.files)
    notes = [*apply_result.notes, *lint_notes, *build_notes]
    return apply_result, lint_ok, build_ok, notes


def files_to_map(file_items: Iterable[tuple[str, str]]) -> dict[str, str]:
    return {path: content for path, content in file_items}
