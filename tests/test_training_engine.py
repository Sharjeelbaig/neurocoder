from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from model.config import TinyMoEConfig
from model.tiny_moe import TORCH_AVAILABLE
from train.config import TrainConfig, TrainStage

if TORCH_AVAILABLE:
    import torch
    from train.engine import build_engine


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class TrainingEngineTests(unittest.TestCase):
    def test_dryrun_fit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_config = TinyMoEConfig(
                vocab_size=128,
                hidden_size=64,
                num_layers=2,
                num_heads=4,
                num_experts=4,
                context_length=64,
            )
            train_config = TrainConfig(
                stage=TrainStage.PRETRAIN,
                max_steps=3,
                save_every=2,
                output_dir=tmp,
                collapse_patience=2,
            )
            engine = build_engine(model_config, train_config, Path(tmp))

            def batches():
                for _ in range(3):
                    ids = torch.randint(0, model_config.vocab_size, (2, 16))
                    yield {"input_ids": ids, "labels": ids.clone()}

            metrics = engine.fit(batches())
            self.assertEqual(len(metrics), 3)
            self.assertTrue((Path(tmp) / "metrics.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
