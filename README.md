# NeuroCoder

NeuroCoder is a from-scratch, narrow-domain coding SLM for:

- React + Tailwind landing page generation.
- Patch-first code edit requests (unified diff output).

This repository is organized in batches so implementation can land incrementally across API calls.

## Repository Layout

- `model/`: TinyMoE model architecture (Transformer + sparse MoE FFN).
- `data/`: License-gated ingestion, dedup, classification, synthetic instruction generation.
- `train/`: Tokenizer training, preprocessing, stage-based training engine.
- `infer/`: API contracts, constrained patch decoding, validation/repair runtime.
- `eval/`: Narrow benchmark harness and baseline comparison scorecard.
- `release/`: Quantization and packaging to Hugging Face and Ollama.
- `tests/`: Unit tests for contracts, diff logic, ingestion, and runtime behavior.

## Quick Start

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/run_api.py --host 127.0.0.1 --port 8080
```

Then call:

```bash
curl -X POST http://127.0.0.1:8080/v1/task \
  -H 'Content-Type: application/json' \
  -d '{
    "task_type": "patch_edit",
    "instruction": "change button color to emerald in Hero component",
    "files": [{"path": "src/Hero.tsx", "content": "export default function Hero(){return <button className=\"bg-blue-500 text-white\">Start</button>}"}],
    "constraints": {"framework": "react-tailwind", "output": "unified_diff"}
  }'
```

## Status

This is a fast-track implementation scaffold for batches 1-10. The interfaces, validators, data pipeline, and runtime are implemented and testable. GPU-scale training, large dataset acquisition, and external benchmark runs require infrastructure and credentials.
