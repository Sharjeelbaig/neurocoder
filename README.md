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

## Batch Commands

```bash
# vNext long session (recommended)
python3 scripts/run_vnext_session.py --profile fast
python3 scripts/run_vnext_session.py --execute --profile full

# See full runbook:
# docs/NEUROCODER_VNEXT_SESSION.md

# Build ground-up dataset with privacy scrub (default ban-term: inferencia)
python3 scripts/build_groundup_dataset_v3.py --landing-count 1200 --patch-count 900 --reasoning-count 300

# Build synthetic curriculum for coding + chat alignment
python3 scripts/build_curriculum.py --out datasets/curriculum/coding_chat_v1.txt

# Train from scratch (base MoE checkpoint)
python3 scripts/train_from_scratch.py \
  --source-dir datasets/curriculum \
  --include-datasets \
  --out-dir artifacts/trained_base \
  --hidden-size 256 --num-layers 8 --num-heads 8 --num-experts 4 \
  --steps 500 --seq-len 256 --batch-size 6

# Build and run high-quality instruction alignment
python3 scripts/build_alignment_set.py --out datasets/curriculum/alignment_v2.txt --repeats 800
python3 scripts/align_responses.py --model-dir artifacts/trained_sft_v3 --dataset datasets/curriculum/alignment_v2.txt --steps 1200

# Optional: blend + slice external HF datasets (Tailwind + reasoning)
python3 scripts/build_hf_mix_dataset.py \
  --out datasets/curriculum/hf_mix_v1.txt \
  --manifest datasets/curriculum/hf_mix_v1_manifest.json \
  --per-dataset 600
python3 scripts/align_responses.py \
  --model-dir artifacts/trained_sft_v3 \
  --dataset datasets/curriculum/hf_mix_v1.txt \
  --steps 520

# Inference CLI (with deterministic quality fallback enabled by default)
python3 scripts/infer_neurocoder.py --model-dir artifacts/release_sft_v3/hf --prompt "generate a landing page"

# Raw output path (fallback disabled, stability guards still active)
python3 scripts/infer_neurocoder.py --model-dir artifacts/release_sft_v3/hf --prompt "generate a landing page" --disable-fallback

# Regression suite for inference quality (runs fallback + disable-fallback modes)
python3 scripts/test_inference_suite.py --model-dir artifacts/release_hfstd_v2/hf

# Batch 2: ingest data with license gate
python3 scripts/run_ingest.py /path/to/source-repo --out datasets/snapshot_v1

# Batch 3: train tokenizer from scratch
python3 scripts/train_tokenizer.py datasets/snapshot_v1/manifest.jsonl --out artifacts/tokenizer/tokenizer.json

# Batch 6: synthetic instruction generation
python3 scripts/gen_synthetic.py datasets/snapshot_v1/manifest.jsonl --out datasets/synthetic/sft_color_edits.jsonl

# Batch 8: run narrow benchmark sample
python3 scripts/run_benchmark.py benchmarks/suites/narrowcoder_v1_sample.jsonl --out benchmarks/results/local

# Batch 9-10: package for HF and Ollama (model name must be neurocoder)
python3 scripts/package_release.py \
  --tokenizer artifacts/trained_sft_v3/tokenizer.json \
  --weights artifacts/trained_sft_v3/model.safetensors \
  --model-config artifacts/trained_sft_v3/model_config.json \
  --out artifacts/release_sft_v3 \
  --model-name neurocoder

# Standard Transformers usage (works with trust_remote_code)
python3 - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = "Sharjeelbaig/neurocoder"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
prompt = "Write a python function to reverse a string"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=120,
    do_sample=False,
    repetition_penalty=1.22,
    no_repeat_ngram_size=6,
    use_cache=True,
)
text = tok.decode(out[0], skip_special_tokens=True)
print(text.split("\nAssistant:", 1)[-1].strip())
PY

# Optional pipeline usage
python3 - <<'PY'
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
model_id = "Sharjeelbaig/neurocoder"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
pipe = pipeline("text-generation", model=model, tokenizer=tok)
print(pipe("Generate a landing page for marketing agency", max_new_tokens=120, do_sample=True, temperature=0.7)[0]["generated_text"])
PY

# Generic CLI for your own prompts (standard HF API)
python3 scripts/infer_hf_standard.py --model-id Sharjeelbaig/neurocoder --prompt "hi"
```
