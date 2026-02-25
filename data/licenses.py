"""SPDX license detection and policy checks."""

from __future__ import annotations

from pathlib import Path
import re

ALLOWED_SPDX = {
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "Apache-2.0",
    "CC0-1.0",
    "CC-BY-4.0",
}

_LICENSE_HINTS: list[tuple[str, str]] = [
    ("MIT License", "MIT"),
    ("Apache License", "Apache-2.0"),
    ("BSD 3-Clause", "BSD-3-Clause"),
    ("BSD 2-Clause", "BSD-2-Clause"),
    ("Creative Commons Zero", "CC0-1.0"),
    ("Creative Commons Attribution", "CC-BY-4.0"),
]

_SPDX_RE = re.compile(r"SPDX-License-Identifier:\s*([A-Za-z0-9\-.+]+)")


def detect_spdx(repo_root: Path) -> str | None:
    """Detect SPDX identifier using repo metadata and license files."""
    package_json = repo_root / "package.json"
    if package_json.exists():
        try:
            text = package_json.read_text(encoding="utf-8", errors="ignore")
            match = re.search(r'"license"\s*:\s*"([^"]+)"', text)
            if match:
                return match.group(1)
        except OSError:
            pass

    for license_name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
        path = repo_root / license_name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")

        spdx_match = _SPDX_RE.search(text)
        if spdx_match:
            return spdx_match.group(1)

        for hint, spdx in _LICENSE_HINTS:
            if hint.lower() in text.lower():
                return spdx

    return None


def is_license_allowed(spdx_id: str | None) -> bool:
    return bool(spdx_id and spdx_id in ALLOWED_SPDX)
