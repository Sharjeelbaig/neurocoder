"""Quantization helpers for deployment packaging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore[assignment]


@dataclass(slots=True)
class QuantizeReport:
    input_path: str
    output_path: str
    method: str
    bytes_written: int



def quantize_array_to_int4(array: "np.ndarray") -> tuple[bytes, float]:
    if np is None:
        raise RuntimeError("numpy is required for int4 quantization")
    max_abs = float(np.max(np.abs(array))) if array.size else 1.0
    scale = max(max_abs / 7.0, 1e-8)
    q = np.clip(np.round(array / scale), -8, 7).astype(np.int8)

    # Pack two signed 4-bit values per byte.
    packed = bytearray()
    if len(q) % 2 != 0:
        q = np.concatenate([q, np.array([0], dtype=np.int8)])

    for i in range(0, len(q), 2):
        lo = int(q[i] & 0x0F)
        hi = int(q[i + 1] & 0x0F)
        packed.append((hi << 4) | lo)

    return bytes(packed), scale



def write_dummy_gguf(output_path: Path, model_name: str = "tinymoe-coder") -> QuantizeReport:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        f"GGUF_PLACEHOLDER\nmodel={model_name}\n"
        "This file is a placeholder. Replace with real quantized weights after training.\n"
    ).encode("utf-8")
    output_path.write_bytes(payload)
    return QuantizeReport(
        input_path="N/A",
        output_path=str(output_path),
        method="placeholder",
        bytes_written=len(payload),
    )
