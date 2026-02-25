"""Run a tiny synthetic training dry run for sanity checks."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.config import TinyMoEConfig
from model.tiny_moe import TORCH_AVAILABLE
from train.config import TrainConfig, TrainStage
from train.engine import build_engine

if TORCH_AVAILABLE:
    import torch



def main() -> None:
    if not TORCH_AVAILABLE:
        raise SystemExit("PyTorch not available. Install with: pip install .[train]")

    model_config = TinyMoEConfig(
        vocab_size=512,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        num_experts=4,
        context_length=128,
    )
    train_config = TrainConfig(
        stage=TrainStage.PRETRAIN,
        max_steps=8,
        save_every=4,
        output_dir="artifacts/dryrun",
        collapse_patience=4,
    )
    engine = build_engine(model_config, train_config, Path(train_config.output_dir))

    def batches():
        for _ in range(train_config.max_steps):
            input_ids = torch.randint(0, model_config.vocab_size, (2, 64))
            labels = input_ids.clone()
            yield {"input_ids": input_ids, "labels": labels}

    metrics = engine.fit(batches())
    print(f"completed {len(metrics)} steps")


if __name__ == "__main__":
    main()
