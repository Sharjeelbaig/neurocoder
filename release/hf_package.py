"""Hugging Face artifact packager for NeuroCoder releases."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

from safetensors import safe_open
from safetensors.torch import load_file, save_file


@dataclass(slots=True)
class HFPackageResult:
    output_dir: str
    files_written: list[str]


def _copy_weights_with_pt_metadata(src: Path, dst: Path) -> None:
    metadata: dict[str, str] = {}
    with safe_open(str(src), framework="pt") as handle:
        metadata = dict(handle.metadata() or {})

    if metadata.get("format") == "pt":
        shutil.copy2(src, dst)
        return

    tensors = load_file(str(src))
    metadata["format"] = "pt"
    save_file(tensors, str(dst), metadata=metadata)



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

    tokenizer_payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    vocab = tokenizer_payload.get("vocab", {})
    pad_id = int(vocab.get("<pad>", 0))
    bos_id = int(vocab.get("<bos>", 1))
    eos_id = int(vocab.get("<eos>", 2))
    unk_id = int(vocab.get("<unk>", 3))

    hf_config = dict(model_config)
    if "top_k" in hf_config and "router_top_k" not in hf_config:
        hf_config["router_top_k"] = hf_config.pop("top_k")
    hf_config["model_type"] = "neurocoder"
    hf_config["architectures"] = ["NeuroCoderForCausalLM"]
    hf_config["use_cache"] = bool(hf_config.get("use_cache", True))
    hf_config["bos_token_id"] = bos_id
    hf_config["eos_token_id"] = eos_id
    hf_config["pad_token_id"] = pad_id
    hf_config["unk_token_id"] = unk_id
    hf_config["auto_map"] = {
        "AutoConfig": "configuration_neurocoder.NeuroCoderConfig",
        "AutoModelForCausalLM": "modeling_neurocoder.NeuroCoderForCausalLM",
        "AutoTokenizer": [
            "tokenization_neurocoder.NeuroCoderTokenizer",
            None,
        ],
    }

    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(hf_config, indent=2, sort_keys=True), encoding="utf-8")
    files_written.append(config_path.name)

    tokenizer_out = output_dir / "tokenizer.json"
    tokenizer_payload.setdefault("added_tokens", [])
    tokenizer_out.write_text(json.dumps(tokenizer_payload, indent=2, sort_keys=True), encoding="utf-8")
    files_written.append(tokenizer_out.name)

    tokenizer_cfg = output_dir / "tokenizer_config.json"
    tokenizer_cfg.write_text(
        json.dumps(
            {
                "model_max_length": int(model_config.get("context_length", 4096)),
                "padding_side": "right",
                "truncation_side": "right",
                "auto_map": {
                    "AutoTokenizer": [
                        "tokenization_neurocoder.NeuroCoderTokenizer",
                        None,
                    ],
                },
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

    generation_cfg = output_dir / "generation_config.json"
    generation_cfg.write_text(
        json.dumps(
            {
                "do_sample": True,
                "temperature": 0.25,
                "top_p": 0.9,
                "repetition_penalty": 1.22,
                "no_repeat_ngram_size": 6,
                "max_new_tokens": 320,
                "use_cache": True,
                "eos_token_id": eos_id,
                "pad_token_id": eos_id,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    files_written.append(generation_cfg.name)

    special_tokens_path = output_dir / "special_tokens_map.json"
    special_tokens_path.write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    files_written.append(special_tokens_path.name)

    has_trained_weights = bool(model_weights and model_weights.exists())
    if has_trained_weights:
        _copy_weights_with_pt_metadata(model_weights, output_dir / "model.safetensors")
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
        f"{weights_note}\n\n"
        "## Transformers Usage\n\n"
        "```python\n"
        "from transformers import AutoTokenizer, AutoModelForCausalLM\n"
        "\n"
        "model_id = \"Sharjeelbaig/neurocoder\"\n"
        "tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)\n"
        "model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)\n"
        "\n"
        "prompt = \"Generate a landing page for marketing agency titled Velocity Landing\"\n"
        "inputs = tokenizer(prompt, return_tensors=\"pt\")\n"
        "outputs = model.generate(\n"
        "    **inputs,\n"
        "    max_new_tokens=220,\n"
        "    do_sample=False,\n"
        "    repetition_penalty=1.22,\n"
        "    no_repeat_ngram_size=6,\n"
        "    use_cache=True,\n"
        ")\n"
        "text = tokenizer.decode(outputs[0], skip_special_tokens=True)\n"
        "print(text.split(\"\\nAssistant:\", 1)[-1].strip())\n"
        "```\n",
        encoding="utf-8",
    )
    files_written.append(readme.name)

    compat_root = Path(__file__).resolve().parents[1] / "hf_compat"
    compat_files = [
        "configuration_neurocoder.py",
        "modeling_neurocoder.py",
        "tokenization_neurocoder.py",
    ]
    for file_name in compat_files:
        src = compat_root / file_name
        dst = output_dir / file_name
        shutil.copy2(src, dst)
        files_written.append(file_name)

    license_path = output_dir / "LICENSE"
    license_path.write_text(license_text + "\n", encoding="utf-8")
    files_written.append(license_path.name)

    return HFPackageResult(output_dir=str(output_dir), files_written=files_written)
