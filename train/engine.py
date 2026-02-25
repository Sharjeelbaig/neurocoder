"""Training engine with checkpoint/resume and expert-collapse telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable

from model.config import TinyMoEConfig
from model.tiny_moe import TORCH_AVAILABLE, TinyMoEModel
from train.config import TrainConfig
from train.telemetry import ExpertTelemetry

if TORCH_AVAILABLE:
    import torch


@dataclass(slots=True)
class TrainStepMetrics:
    step: int
    loss: float
    aux_loss: float
    z_loss: float
    dropped_tokens: int
    expert_load: list[float]
    collapse_alarm: bool


@dataclass(slots=True)
class TrainState:
    step: int = 0


class TrainingEngine:
    def __init__(self, model: Any, config: TrainConfig, output_dir: Path) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for TrainingEngine. Install with: pip install .[train]")
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state = TrainState()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.telemetry = ExpertTelemetry()
        self.metrics_path = self.output_dir / "metrics.jsonl"

        torch.manual_seed(config.seed)
        random.seed(config.seed)

    def train_step(self, batch: dict[str, Any]) -> TrainStepMetrics:
        self.model.train()
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        output = self.model(input_ids=input_ids, labels=labels)
        loss = output["loss"]

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.state.step += 1

        collapse_alarm, _ = self.telemetry.update(
            output.get("expert_load", []),
            threshold=self.config.collapse_threshold,
            patience=self.config.collapse_patience,
        )

        metrics = TrainStepMetrics(
            step=self.state.step,
            loss=float(loss.detach().cpu().item()),
            aux_loss=float(output.get("aux_loss", torch.tensor(0.0)).detach().cpu().item()),
            z_loss=float(output.get("z_loss", torch.tensor(0.0)).detach().cpu().item()),
            dropped_tokens=int(output.get("dropped_tokens", 0)),
            expert_load=[float(x) for x in output.get("expert_load", [])],
            collapse_alarm=collapse_alarm,
        )

        with self.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(metrics), sort_keys=True) + "\n")

        if self.state.step % self.config.save_every == 0:
            self.save_checkpoint(self.output_dir / f"checkpoint-step-{self.state.step}.pt")

        return metrics

    def fit(self, batches: Iterable[dict[str, Any]]) -> list[TrainStepMetrics]:
        all_metrics: list[TrainStepMetrics] = []
        for batch in batches:
            if self.state.step >= self.config.max_steps:
                break
            metrics = self.train_step(batch)
            all_metrics.append(metrics)
        return all_metrics

    def save_checkpoint(self, path: Path) -> None:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.state.step,
            "config": asdict(self.config),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.state.step = int(payload["step"])



def build_engine(model_config: TinyMoEConfig, train_config: TrainConfig, output_dir: Path) -> TrainingEngine:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for build_engine")
    model = TinyMoEModel(model_config)
    return TrainingEngine(model=model, config=train_config, output_dir=output_dir)
