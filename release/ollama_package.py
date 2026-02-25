"""Ollama packaging helpers for TinyMoE GGUF releases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


@dataclass(slots=True)
class OllamaPackageResult:
    output_dir: str
    files_written: list[str]



def build_ollama_package(
    output_dir: Path,
    gguf_path: Path,
    model_name: str = "tinymoe-coder",
) -> OllamaPackageResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_gguf = output_dir / gguf_path.name
    shutil.copy2(gguf_path, copied_gguf)

    modelfile = output_dir / "Modelfile"
    modelfile.write_text(
        (
            f"FROM ./{copied_gguf.name}\n"
            "TEMPLATE \"\"\"{{ .Prompt }}\"\"\"\n"
            "PARAMETER temperature 0.2\n"
            "PARAMETER top_p 0.9\n"
            "SYSTEM \"You are TinyMoE Coder. Return unified diff for patch edits.\"\n"
        ),
        encoding="utf-8",
    )

    readme = output_dir / "README.md"
    readme.write_text(
        f"# {model_name} (Ollama)\n\n"
        "Build locally with:\n"
        f"`ollama create {model_name} -f Modelfile`\n",
        encoding="utf-8",
    )

    return OllamaPackageResult(
        output_dir=str(output_dir),
        files_written=[copied_gguf.name, modelfile.name, readme.name],
    )
