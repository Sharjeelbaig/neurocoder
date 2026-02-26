"""Standard Transformers inference CLI for NeuroCoder."""

from __future__ import annotations

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_assistant(text: str) -> str:
    marker = "\nAssistant:"
    if marker not in text:
        return text.strip()
    return text.split(marker, 1)[1].strip()


def run_once(
    *,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=1.2,
        no_repeat_ngram_size=4,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        # Avoid transformers warning when generation_config contains sampling-only fields.
        model.generation_config.temperature = None
        model.generation_config.top_p = None

    outputs = model.generate(
        **inputs,
        **gen_kwargs,
    )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extract_assistant(full_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference CLI using standard Transformers API")
    parser.add_argument("--model-id", default="Sharjeelbaig/neurocoder")
    parser.add_argument("--prompt", default="", help="Single prompt. If omitted, starts interactive mode.")
    parser.add_argument("--max-new-tokens", type=int, default=260)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True)

    if args.prompt.strip():
        print(
            run_once(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt.strip(),
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        )
        return

    print("NeuroCoder HF interactive mode. Type 'exit' to quit.")
    while True:
        try:
            prompt = input("prompt> ").strip()
        except EOFError:
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break
        output = run_once(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"output> {output}")


if __name__ == "__main__":
    main()
