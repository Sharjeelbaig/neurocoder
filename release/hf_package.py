"""Hugging Face artifact packager for NeuroCoder releases."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil


@dataclass(slots=True)
class HFPackageResult:
    output_dir: str
    files_written: list[str]



def build_hf_package(
    output_dir: Path,
    tokenizer_json: Path,
    model_config: dict,
    model_weights: Path | None = None,
    license_text: str = "MIT",
    model_name: str = "neurocoder",
) -> HFPackageResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    files_written: list[str] = []

    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(model_config, indent=2, sort_keys=True), encoding="utf-8")
    files_written.append(config_path.name)

    tokenizer_out = output_dir / "tokenizer.json"
    shutil.copy2(tokenizer_json, tokenizer_out)
    files_written.append(tokenizer_out.name)

    tokenizer_cfg = output_dir / "tokenizer_config.json"
    tokenizer_cfg.write_text(
        json.dumps(
            {
                "model_max_length": int(model_config.get("context_length", 4096)),
                "padding_side": "right",
                "truncation_side": "right",
                "special_tokens_map": {
                    "bos_token": "<bos>",
                    "eos_token": "<eos>",
                    "pad_token": "<pad>",
                    "unk_token": "<unk>",
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    files_written.append(tokenizer_cfg.name)

    has_trained_weights = bool(model_weights and model_weights.exists())
    if has_trained_weights:
        shutil.copy2(model_weights, output_dir / "model.safetensors")
    else:
        (output_dir / "model.safetensors").write_bytes(
            b"TINYMOE_PLACEHOLDER\nreplace-with-trained-safetensors\n"
        )
    files_written.append("model.safetensors")

    readme = output_dir / "README.md"
    display_name = model_name.replace("-", " ").title()
    if has_trained_weights:
        weights_note = "Includes trained `model.safetensors` weights."
    else:
        weights_note = "Contains placeholder weights; replace `model.safetensors` with trained weights."
    readme.write_text(
        "---\n"
        "license: mit\n"
        "language:\n"
        "- en\n"
        "tags:\n"
        "- code\n"
        "- moe\n"
        "- react\n"
        "- tailwind\n"
        "library_name: pytorch\n"
        "---\n\n"
        f"# {display_name}\n\n"
        "From-scratch narrow-domain coding SLM for React + Tailwind generation and unified-diff edits.\n\n"
        f"{weights_note}\n",
        encoding="utf-8",
    )
    files_written.append(readme.name)

    license_path = output_dir / "LICENSE"
    license_path.write_text(license_text + "\n", encoding="utf-8")
    files_written.append(license_path.name)

    return HFPackageResult(output_dir=str(output_dir), files_written=files_written)
