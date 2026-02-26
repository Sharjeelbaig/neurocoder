"""Build Hugging Face and Ollama package folders from local artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

from release.hf_package import build_hf_package
from release.ollama_package import build_ollama_package
from release.quantize import write_dummy_gguf

def main() -> None:
    parser = argparse.ArgumentParser(description="Package NeuroCoder for HF and Ollama")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    parser.add_argument("--weights", default="", help="Path to model.safetensors")
    parser.add_argument("--model-config", default="", help="Path to model config JSON")
    parser.add_argument("--out", default="artifacts/release")
    parser.add_argument("--model-name", default="neurocoder")
    args = parser.parse_args()

    out_dir = Path(args.out)
    hf_dir = out_dir / "hf"
    ollama_dir = out_dir / "ollama"

    default_model_config = {
        "architectures": ["TinyMoEModel"],
        "model_type": "tinymoe",
        "context_length": 4096,
        "vocab_size": 32000,
        "num_experts": 8,
        "top_k": 2,
    }
    model_config = default_model_config
    if args.model_config:
        config_path = Path(args.model_config)
        model_config = json.loads(config_path.read_text(encoding="utf-8"))

    weights_path = Path(args.weights) if args.weights else None
    hf_result = build_hf_package(
        output_dir=hf_dir,
        tokenizer_json=Path(args.tokenizer),
        model_config=model_config,
        model_weights=weights_path,
        license_text=Path("LICENSE").read_text(encoding="utf-8"),
        model_name=args.model_name,
    )

    gguf_path = out_dir / f"{args.model_name}.Q4_K_M.gguf"
    write_dummy_gguf(gguf_path, model_name=args.model_name)
    ollama_result = build_ollama_package(
        output_dir=ollama_dir,
        gguf_path=gguf_path,
        model_name=args.model_name,
    )

    print(hf_result)
    print(ollama_result)


if __name__ == "__main__":
    main()
