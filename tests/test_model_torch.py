from __future__ import annotations

import unittest

from model.config import TinyMoEConfig
from model.tiny_moe import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    from model.tiny_moe import TinyMoEModel


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class ModelTorchTests(unittest.TestCase):
    def test_forward_shape(self) -> None:
        config = TinyMoEConfig(
            vocab_size=128,
            context_length=64,
            hidden_size=64,
            num_layers=4,
            num_heads=4,
            num_experts=4,
        )
        model = TinyMoEModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels)
        self.assertIn("logits", out)
        self.assertIn("loss", out)
        self.assertEqual(out["logits"].shape, (2, 16, config.vocab_size))


if __name__ == "__main__":
    unittest.main()
