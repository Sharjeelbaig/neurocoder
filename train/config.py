"""Stage-wise training configuration and TOML loading."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import tomllib


class TrainStage(str, Enum):
    PRETRAIN = "pretrain"
    SFT = "sft"
    PREFERENCE = "preference"


@dataclass(slots=True)
class TrainConfig:
    stage: TrainStage
    seed: int = 42
    global_batch_size: int = 8
    micro_batch_size: int = 2
    learning_rate: float = 3e-4
    warmup_steps: int = 100
    max_steps: int = 1000
    log_every: int = 10
    save_every: int = 100
    output_dir: str = "artifacts/checkpoints"
    world_size: int = 1
    collapse_threshold: float = 0.02
    collapse_patience: int = 200



def load_config(path: Path) -> TrainConfig:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    stage = TrainStage(payload["stage"])
    return TrainConfig(
        stage=stage,
        seed=int(payload.get("seed", 42)),
        global_batch_size=int(payload.get("global_batch_size", 8)),
        micro_batch_size=int(payload.get("micro_batch_size", 2)),
        learning_rate=float(payload.get("learning_rate", 3e-4)),
        warmup_steps=int(payload.get("warmup_steps", 100)),
        max_steps=int(payload.get("max_steps", 1000)),
        log_every=int(payload.get("log_every", 10)),
        save_every=int(payload.get("save_every", 100)),
        output_dir=str(payload.get("output_dir", "artifacts/checkpoints")),
        world_size=int(payload.get("world_size", 1)),
        collapse_threshold=float(payload.get("collapse_threshold", 0.02)),
        collapse_patience=int(payload.get("collapse_patience", 200)),
    )
