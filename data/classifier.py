"""Classifiers for React+Tailwind targeted corpus filtering."""

from __future__ import annotations

from pathlib import Path
import re

_REACT_HINTS = [
    "from 'react'",
    'from "react"',
    "function",
    "return (",
    "className=",
]

_TAILWIND_TOKEN_RE = re.compile(r"\b(?:bg|text|border|px|py|mx|my|grid|flex|rounded|shadow)-[a-z0-9\-/]+")


def is_code_file(path: Path) -> bool:
    return path.suffix.lower() in {
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".css",
        ".html",
        ".json",
        ".md",
    }


def classify_react_tailwind(content: str) -> dict[str, bool]:
    lowered = content.lower()
    react_like = sum(1 for hint in _REACT_HINTS if hint.lower() in lowered) >= 2
    tailwind_like = bool(_TAILWIND_TOKEN_RE.search(content)) or "tailwind" in lowered
    return {
        "react_like": react_like,
        "tailwind_like": tailwind_like,
        "domain_match": react_like and tailwind_like,
    }
