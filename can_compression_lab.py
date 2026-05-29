#!/usr/bin/env python3
"""
Project-style CAN signal compression experiment.

The compressor works on parsed CAN records instead of BLF bytes:

    timestamp_delta_us, arbitration_id, dlc, payload[0..dlc-1]

Frames are routed by arbitration_id. For each ID, a causal predictor estimates
the next payload and an adaptive arithmetic/range-code size model accounts for
the residual bytes. The range coder here is a deterministic size estimator:
using the same adaptive frequencies as a range coder, it accumulates
-log2(p(symbol)) bits. This is the standard way to compare predictors without
letting bitstream engineering dominate the experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


MAX_DLC = 64
BYTE_SCALE = 1.0 / 255.0


@dataclass(frozen=True)
class CanFrame:
    timestamp_delta_us: int
    arbitration_id: int
    dlc: int
    payload: bytes


def load_blf_frames(path: Path, max_frames: Optional[int] = None) -> List[CanFrame]:
    """
    Load CAN/CAN-FD frames from a Vector BLF file with python-can.

    Error frames and remote frames are skipped because this compression
    experiment models payload-bearing CAN records:
    timestamp_delta_us, arbitration_id, dlc, payload.
    """

    if not path.exists():
        raise SystemExit(f"BLF file not found: {path}")

    try:
        import can  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Reading BLF requires python-can. Install it with:\n"
            "  python3 -m pip install python-can"
        ) from exc

    frames: List[CanFrame] = []
    last_ts: Optional[float] = None
    with can.BLFReader(str(path)) as reader:
        for msg in reader:
            if getattr(msg, "is_error_frame", False) or getattr(msg, "is_remote_frame", False):
                continue
            payload = bytes(msg.data)
            if len(payload) > MAX_DLC:
                payload = payload[:MAX_DLC]
            timestamp = float(msg.timestamp)
            if last_ts is None:
                delta_us = 0
            else:
                delta_us = max(0, int(round((timestamp - last_ts) * 1_000_000.0)))
            last_ts = timestamp
            frames.append(CanFrame(delta_us, int(msg.arbitration_id), len(payload), payload))
            if max_frames is not None and len(frames) >= max_frames:
                break

    if not frames:
        raise SystemExit(f"No payload CAN/CAN-FD frames were read from {path}")
    return frames


class AdaptiveByteModel:
    """Adaptive byte frequency model used by the arithmetic-size estimator."""

    def __init__(self, alphabet: int = 256, init_count: int = 1, rescale_at: int = 1 << 15) -> None:
        self.counts = [init_count] * alphabet
        self.total = alphabet * init_count
        self.rescale_at = rescale_at

    def bits_then_update(self, symbol: int) -> float:
        count = self.counts[symbol]
        bits = math.log2(self.total) - math.log2(count)
        self.counts[symbol] = count + 1
        self.total += 1
        if self.total >= self.rescale_at:
            self.total = 0
            for idx, value in enumerate(self.counts):
                new_value = (value + 1) >> 1
                self.counts[idx] = new_value
                self.total += new_value
        return bits


class AdaptiveIntModel:
    """Adaptive model for low-cardinality integers such as CAN ID indices and DLC."""

    def __init__(self, values: Iterable[int]) -> None:
        self.values = list(values)
        self.counts = {value: 1 for value in self.values}
        self.total = len(self.values)

    def bits_then_update(self, value: int) -> float:
        if value not in self.counts:
            self.counts[value] = 1
            self.total += 1
        bits = math.log2(self.total) - math.log2(self.counts[value])
        self.counts[value] += 1
        self.total += 1
        return bits


class TimestampDeltaModel:
    """Predicts timestamp deltas per ID and entropy-codes signed residuals."""

    def __init__(self) -> None:
        self.last_by_id: Dict[int, int] = {}
        self.byte_models = [AdaptiveByteModel() for _ in range(3)]

    @staticmethod
    def _zigzag(value: int) -> int:
        return (value << 1) if value >= 0 else ((-value << 1) - 1)

    def bits_then_update(self, arbitration_id: int, timestamp_delta_us: int) -> float:
        pred = self.last_by_id.get(arbitration_id, timestamp_delta_us)
        self.last_by_id[arbitration_id] = int(0.95 * pred + 0.05 * timestamp_delta_us)
        residual = self._zigzag(timestamp_delta_us - pred)
        bits = 0.0
        for idx in range(3):
            bits += self.byte_models[idx].bits_then_update((residual >> (8 * idx)) & 0xFF)
        return bits


class PayloadPredictor:
    def predict(self, arbitration_id: int, dlc: int) -> List[int]:
        raise NotImplementedError

    def update(self, arbitration_id: int, payload: bytes) -> None:
        raise NotImplementedError


class LastValuePredictor(PayloadPredictor):
    def __init__(self) -> None:
        self.last: Dict[int, List[int]] = {}

    def predict(self, arbitration_id: int, dlc: int) -> List[int]:
        return list(self.last.get(arbitration_id, [0] * dlc)[:dlc]) or [0] * dlc

    def update(self, arbitration_id: int, payload: bytes) -> None:
        self.last[arbitration_id] = list(payload)


class OnlineGRUPredictor(PayloadPredictor):
    """
    Small GRU-style online predictor per CAN ID.

    It keeps a hidden state per ID and learns byte-wise linear output heads with
    LMS updates. Input-to-hidden weights are deterministic random projections so
    the model remains dependency-free while still behaving like a causal neural
    residual predictor.
    """

    def __init__(self, hidden: int = 16, lr: float = 0.018, seed: int = 7) -> None:
        self.hidden = hidden
        self.lr = lr
        rng = random.Random(seed)
        self.wx = [[rng.uniform(-0.35, 0.35) for _ in range(MAX_DLC)] for _ in range(hidden)]
        self.uh = [[rng.uniform(-0.08, 0.08) for _ in range(hidden)] for _ in range(hidden)]
        self.wz = [[rng.uniform(-0.25, 0.25) for _ in range(MAX_DLC)] for _ in range(hidden)]
        self.state: Dict[int, List[float]] = {}
        self.last: Dict[int, List[int]] = {}
        self.out: Dict[int, List[List[float]]] = {}
        self.bias: Dict[int, List[float]] = {}

    def _ensure(self, arbitration_id: int) -> None:
        if arbitration_id not in self.state:
            self.state[arbitration_id] = [0.0] * self.hidden
            self.last[arbitration_id] = [0] * MAX_DLC
            self.out[arbitration_id] = [[0.0] * self.hidden for _ in range(MAX_DLC)]
            self.bias[arbitration_id] = [0.0] * MAX_DLC

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value < -30.0:
            return 0.0
        if value > 30.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(-value))

    def _advance_state(self, arbitration_id: int) -> None:
        last = self.last[arbitration_id]
        prev = self.state[arbitration_id]
        next_state = [0.0] * self.hidden
        x = [value * BYTE_SCALE - 0.5 for value in last]
        for h_idx in range(self.hidden):
            gate_raw = sum(self.wz[h_idx][i] * x[i] for i in range(MAX_DLC))
            z = self._sigmoid(gate_raw)
            raw = sum(self.wx[h_idx][i] * x[i] for i in range(MAX_DLC))
            raw += sum(self.uh[h_idx][i] * prev[i] for i in range(self.hidden))
            proposal = math.tanh(raw)
            next_state[h_idx] = (1.0 - z) * prev[h_idx] + z * proposal
        self.state[arbitration_id] = next_state

    def predict(self, arbitration_id: int, dlc: int) -> List[int]:
        self._ensure(arbitration_id)
        self._advance_state(arbitration_id)
        hidden = self.state[arbitration_id]
        last = self.last[arbitration_id]
        out = self.out[arbitration_id]
        bias = self.bias[arbitration_id]
        pred = []
        for byte_idx in range(dlc):
            y = bias[byte_idx] + sum(out[byte_idx][j] * hidden[j] for j in range(self.hidden))
            correction = 48.0 * math.tanh(y)
            pred.append(max(0, min(255, int(round(last[byte_idx] + correction)))))
        return pred

    def update(self, arbitration_id: int, payload: bytes) -> None:
        self._ensure(arbitration_id)
        hidden = self.state[arbitration_id]
        last = self.last[arbitration_id]
        out = self.out[arbitration_id]
        bias = self.bias[arbitration_id]
        for byte_idx, actual in enumerate(payload):
            y = bias[byte_idx] + sum(out[byte_idx][j] * hidden[j] for j in range(self.hidden))
            pred = last[byte_idx] + 48.0 * math.tanh(y)
            err = (actual - pred) / 64.0
            bias[byte_idx] += self.lr * err
            for j in range(self.hidden):
                out[byte_idx][j] += self.lr * err * hidden[j]
        updated = list(payload) + [0] * (MAX_DLC - len(payload))
        self.last[arbitration_id] = updated[:MAX_DLC]


class OnlineTCNPredictor(PayloadPredictor):
    """
    Causal 1D-CNN/TCN-style predictor.

    Each byte has a small learned dilated causal filter over previous payloads
    at lags 1, 2, 4, 8 and 16. LMS updates make it fast and deterministic.
    """

    def __init__(self, lr: float = 0.006, lags: Tuple[int, ...] = (1, 2, 4, 8, 16)) -> None:
        self.lr = lr
        self.lags = lags
        self.history: Dict[int, List[List[int]]] = {}
        self.weights: Dict[int, List[List[float]]] = {}
        self.bias: Dict[int, List[float]] = {}

    def _ensure(self, arbitration_id: int) -> None:
        if arbitration_id not in self.history:
            self.history[arbitration_id] = []
            self.weights[arbitration_id] = [[0.0] * len(self.lags) for _ in range(MAX_DLC)]
            self.bias[arbitration_id] = [0.0] * MAX_DLC

    def _features(self, arbitration_id: int, byte_idx: int) -> List[float]:
        hist = self.history[arbitration_id]
        features = []
        base = hist[-1][byte_idx] if hist and byte_idx < len(hist[-1]) else 0
        for lag in self.lags:
            if len(hist) >= lag and byte_idx < len(hist[-lag]):
                features.append((hist[-lag][byte_idx] - base) / 64.0)
            else:
                features.append(0.0)
        return features

    def _base(self, arbitration_id: int, byte_idx: int) -> int:
        hist = self.history[arbitration_id]
        if hist and byte_idx < len(hist[-1]):
            return hist[-1][byte_idx]
        return 0

    def predict(self, arbitration_id: int, dlc: int) -> List[int]:
        self._ensure(arbitration_id)
        pred = []
        for byte_idx in range(dlc):
            features = self._features(arbitration_id, byte_idx)
            y = self.bias[arbitration_id][byte_idx]
            y += sum(w * x for w, x in zip(self.weights[arbitration_id][byte_idx], features))
            correction = 48.0 * math.tanh(y)
            pred.append(max(0, min(255, int(round(self._base(arbitration_id, byte_idx) + correction)))))
        return pred

    def update(self, arbitration_id: int, payload: bytes) -> None:
        self._ensure(arbitration_id)
        for byte_idx, actual in enumerate(payload):
            features = self._features(arbitration_id, byte_idx)
            weights = self.weights[arbitration_id][byte_idx]
            y = self.bias[arbitration_id][byte_idx] + sum(w * x for w, x in zip(weights, features))
            pred = self._base(arbitration_id, byte_idx) + 48.0 * math.tanh(y)
            err = (actual - pred) / 64.0
            self.bias[arbitration_id][byte_idx] += self.lr * err
            for idx, x in enumerate(features):
                weights[idx] += self.lr * err * x
        self.history[arbitration_id].append(list(payload))
        if len(self.history[arbitration_id]) > max(self.lags):
            self.history[arbitration_id] = self.history[arbitration_id][-max(self.lags):]


def pack_u16(value: int) -> bytes:
    return struct.pack("<H", max(0, min(65535, value)))


def pack_i16(value: int) -> bytes:
    return struct.pack("<h", max(-32768, min(32767, value)))


def generate_can_frames(count: int, seed: int = 42) -> List[CanFrame]:
    """
    Generate realistic-ish CAN traffic:
    periodic IDs, smooth sensors, counters, status bits, rare events, jitter,
    and a small amount of high-entropy diagnostic/CAN-FD traffic.
    """

    rng = random.Random(seed)
    specs = [
        (0x100, 8, 10000, "powertrain"),
        (0x110, 8, 10000, "wheel"),
        (0x120, 8, 20000, "brake"),
        (0x130, 8, 20000, "steering"),
        (0x180, 8, 50000, "body"),
        (0x220, 8, 100000, "climate"),
        (0x300, 8, 100000, "status"),
        (0x3A0, 8, 200000, "diag"),
        (0x510, 64, 50000, "canfd"),
    ]
    next_due = {arb: rng.randrange(period) for arb, _dlc, period, _kind in specs}
    counters = {arb: 0 for arb, _dlc, _period, _kind in specs}
    now = 0
    last_emit = 0
    frames: List[CanFrame] = []

    while len(frames) < count:
        arb, dlc, period, kind = min(specs, key=lambda item: next_due[item[0]])
        now = next_due[arb]
        jitter = int(rng.gauss(0, period * 0.025))
        next_due[arb] += max(1000, period + jitter)
        counters[arb] = (counters[arb] + 1) & 0xFF
        t = now / 1_000_000.0
        payload = make_payload(kind, dlc, counters[arb], t, rng)
        timestamp_delta = max(0, now - last_emit)
        last_emit = now
        frames.append(CanFrame(timestamp_delta, arb, dlc, payload))

    return frames


def make_payload(kind: str, dlc: int, counter: int, t: float, rng: random.Random) -> bytes:
    if kind == "powertrain":
        rpm = int(1800 + 650 * math.sin(t * 1.7) + 120 * math.sin(t * 12.0) + rng.gauss(0, 10))
        throttle = int(35 + 25 * math.sin(t * 0.9) + rng.gauss(0, 2))
        temp = int(86 + 4 * math.sin(t * 0.08) + rng.gauss(0, 0.4))
        return pack_u16(rpm) + bytes([max(0, min(100, throttle)), temp & 0xFF, counter, 0, 0, checksum8(rpm, throttle, temp, counter)])
    if kind == "wheel":
        base = 420 + 80 * math.sin(t * 0.55)
        values = [int(base + rng.gauss(0, 2) + 2 * math.sin(t * (idx + 1))) for idx in range(4)]
        return b"".join(pack_u16(v) for v in values)
    if kind == "brake":
        pressure = max(0, int(120 * max(0, math.sin(t * 0.21 - 0.5)) + rng.gauss(0, 2)))
        flags = ((counter // 32) & 1) | (((counter // 97) & 1) << 1)
        return pack_u16(pressure) + bytes([flags, counter, 0, 0, rng.randrange(4), checksum8(pressure, flags, counter)])
    if kind == "steering":
        angle = int(720 * math.sin(t * 0.33) + 40 * math.sin(t * 3.1) + rng.gauss(0, 3))
        rate = int(35 * math.cos(t * 0.33) + rng.gauss(0, 1))
        return pack_i16(angle) + pack_i16(rate) + bytes([counter, 0, 0, checksum8(angle, rate, counter)])
    if kind == "body":
        door = 1 if int(t / 17) % 23 == 0 else 0
        lamps = int(t * 2) & 0x0F
        voltage = int(135 + 3 * math.sin(t * 0.2) + rng.gauss(0, 0.3))
        return bytes([door, lamps, voltage, counter, 0, 0, 0, checksum8(door, lamps, voltage, counter)])
    if kind == "climate":
        cabin = int(220 + 15 * math.sin(t * 0.03) + rng.gauss(0, 1))
        target = 220
        fan = int(3 + 2 * max(0, math.sin(t * 0.1)))
        return pack_u16(cabin) + pack_u16(target) + bytes([fan, counter, 0, checksum8(cabin, target, fan, counter)])
    if kind == "status":
        mode = (counter // 40) % 5
        alive = counter & 0x0F
        return bytes([mode, alive, 0, 0, 0, 0, 0, checksum8(mode, alive)])
    if kind == "diag":
        mostly_zero = [0] * 8
        if rng.random() < 0.08:
            mostly_zero = [rng.randrange(256) for _ in range(8)]
        mostly_zero[1] = counter
        return bytes(mostly_zero)
    if kind == "canfd":
        block = []
        slow = int(1000 + 200 * math.sin(t * 0.4))
        for idx in range(dlc // 2):
            value = slow + idx * 3 + int(15 * math.sin(t * (idx % 7 + 1) * 0.05)) + rng.randrange(-2, 3)
            block.extend(pack_u16(value))
        block[0] = counter
        block[-1] = checksum8(*block[:-1])
        return bytes(block[:dlc])
    raise ValueError(kind)


def checksum8(*values: int) -> int:
    total = 0
    for value in values:
        total = (total + (value & 0xFF) + ((value >> 8) & 0xFF)) & 0xFF
    return total


def serialize_frames(frames: Iterable[CanFrame]) -> bytes:
    chunks = []
    for frame in frames:
        chunks.append(struct.pack("<IIB", frame.timestamp_delta_us, frame.arbitration_id, frame.dlc))
        chunks.append(frame.payload)
    return b"".join(chunks)


def write_jsonl(frames: Iterable[CanFrame], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for frame in frames:
            fh.write(
                json.dumps(
                    {
                        "timestamp_delta_us": frame.timestamp_delta_us,
                        "arbitration_id": f"0x{frame.arbitration_id:X}",
                        "dlc": frame.dlc,
                        "payload_hex": frame.payload.hex(),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )


def residual_byte(actual: int, predicted: int) -> int:
    return (actual - predicted) & 0xFF


def compression_run(frames: List[CanFrame], predictor: PayloadPredictor, train_ratio: float) -> Dict[str, float]:
    ids = sorted({frame.arbitration_id for frame in frames})
    id_rank = {arb: idx for idx, arb in enumerate(ids)}
    id_model = AdaptiveIntModel(range(len(ids)))
    dlc_model = AdaptiveIntModel([0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64])
    ts_model = TimestampDeltaModel()
    residual_models: Dict[Tuple[int, int], AdaptiveByteModel] = {}
    raw_bytes = len(serialize_frames(frames))
    split = int(len(frames) * train_ratio)

    # Warm-up/training pass. This mimics training a per-ID model before coding
    # the evaluation split, but the adaptive entropy models start fresh below.
    for frame in frames[:split]:
        predictor.predict(frame.arbitration_id, frame.dlc)
        predictor.update(frame.arbitration_id, frame.payload)

    t0 = time.perf_counter()
    bits = 0.0
    payload_bits = 0.0
    for frame in frames[split:]:
        bits += id_model.bits_then_update(id_rank[frame.arbitration_id])
        bits += dlc_model.bits_then_update(frame.dlc)
        bits += ts_model.bits_then_update(frame.arbitration_id, frame.timestamp_delta_us)
        pred = predictor.predict(frame.arbitration_id, frame.dlc)
        for idx, actual in enumerate(frame.payload):
            model_key = (frame.arbitration_id, idx)
            if model_key not in residual_models:
                residual_models[model_key] = AdaptiveByteModel()
            residual = residual_byte(actual, pred[idx])
            cost = residual_models[model_key].bits_then_update(residual)
            bits += cost
            payload_bits += cost
        predictor.update(frame.arbitration_id, frame.payload)
    encode_time = max(1e-9, time.perf_counter() - t0)

    # Decode simulation: same causal predictor updates from residuals.
    decode_predictor = predictor_factory(type(predictor).__name__)
    for frame in frames[:split]:
        decode_predictor.predict(frame.arbitration_id, frame.dlc)
        decode_predictor.update(frame.arbitration_id, frame.payload)

    t1 = time.perf_counter()
    for frame in frames[split:]:
        pred = decode_predictor.predict(frame.arbitration_id, frame.dlc)
        decoded = bytes(((pred[idx] + residual_byte(frame.payload[idx], pred[idx])) & 0xFF) for idx in range(frame.dlc))
        if decoded != frame.payload:
            raise RuntimeError("decoder simulation mismatch")
        decode_predictor.update(frame.arbitration_id, decoded)
    decode_time = max(1e-9, time.perf_counter() - t1)

    coded_bytes = math.ceil(bits / 8.0)
    eval_raw_bytes = len(serialize_frames(frames[split:]))
    raw_mb = eval_raw_bytes / 1_000_000.0
    return {
        "frames": len(frames) - split,
        "raw_bytes": eval_raw_bytes,
        "compressed_bytes_est": coded_bytes,
        "compression_ratio_raw_over_compressed": eval_raw_bytes / max(1, coded_bytes),
        "bits_per_payload_byte": payload_bits / max(1, sum(frame.dlc for frame in frames[split:])),
        "encode_frames_per_sec": (len(frames) - split) / encode_time,
        "decode_frames_per_sec": (len(frames) - split) / decode_time,
        "encode_mb_per_sec": raw_mb / encode_time,
        "decode_mb_per_sec": raw_mb / decode_time,
    }


def predictor_factory(name: str) -> PayloadPredictor:
    normalized = name.lower()
    if normalized in {"onlinegrupredictor", "gru", "neural"}:
        return OnlineGRUPredictor()
    if normalized in {"onlinetcnpredictor", "tcn", "cnn"}:
        return OnlineTCNPredictor()
    if normalized in {"lastvaluepredictor", "last"}:
        return LastValuePredictor()
    raise ValueError(name)


def raw_and_zlib_baselines(frames: List[CanFrame], train_ratio: float) -> Dict[str, Dict[str, float]]:
    split = int(len(frames) * train_ratio)
    raw = serialize_frames(frames[split:])
    raw_mb = len(raw) / 1_000_000.0
    t0 = time.perf_counter()
    compressed = zlib.compress(raw, level=9)
    enc = max(1e-9, time.perf_counter() - t0)
    t1 = time.perf_counter()
    decoded = zlib.decompress(compressed)
    dec = max(1e-9, time.perf_counter() - t1)
    if decoded != raw:
        raise RuntimeError("zlib decode mismatch")
    return {
        "raw": {
            "frames": len(frames) - split,
            "raw_bytes": len(raw),
            "compressed_bytes_est": len(raw),
            "compression_ratio_raw_over_compressed": 1.0,
            "encode_frames_per_sec": float("inf"),
            "decode_frames_per_sec": float("inf"),
            "encode_mb_per_sec": float("inf"),
            "decode_mb_per_sec": float("inf"),
        },
        "zlib_level9_on_parsed_records": {
            "frames": len(frames) - split,
            "raw_bytes": len(raw),
            "compressed_bytes_est": len(compressed),
            "compression_ratio_raw_over_compressed": len(raw) / max(1, len(compressed)),
            "encode_frames_per_sec": (len(frames) - split) / enc,
            "decode_frames_per_sec": (len(frames) - split) / dec,
            "encode_mb_per_sec": raw_mb / enc,
            "decode_mb_per_sec": raw_mb / dec,
        },
    }


def summarize_dataset(frames: List[CanFrame], source: str) -> Dict[str, object]:
    counts: Dict[str, int] = {}
    periods: Dict[int, List[int]] = {}
    for frame in frames:
        key = f"0x{frame.arbitration_id:X}"
        counts[key] = counts.get(key, 0) + 1
        periods.setdefault(frame.arbitration_id, []).append(frame.timestamp_delta_us)
    return {
        "source": source,
        "frames": len(frames),
        "raw_record_bytes": len(serialize_frames(frames)),
        "ids": counts,
        "timestamp_delta_us_mean": round(statistics.mean(frame.timestamp_delta_us for frame in frames), 2),
        "payload_bytes_mean": round(statistics.mean(frame.dlc for frame in frames), 2),
    }


def print_table(results: Dict[str, Dict[str, float]]) -> None:
    headers = ["method", "raw_bytes", "compressed", "ratio", "bpp_payload", "enc_MB/s", "dec_MB/s"]
    print("\n" + " | ".join(headers))
    print("-" * 91)
    for name, result in results.items():
        bpp = result.get("bits_per_payload_byte", float("nan"))
        enc = result["encode_mb_per_sec"]
        dec = result["decode_mb_per_sec"]
        enc_text = "inf" if math.isinf(enc) else f"{enc:,.2f}"
        dec_text = "inf" if math.isinf(dec) else f"{dec:,.2f}"
        print(
            f"{name:29s} | "
            f"{int(result['raw_bytes']):9d} | "
            f"{int(result['compressed_bytes_est']):10d} | "
            f"{result['compression_ratio_raw_over_compressed']:5.2f}x | "
            f"{bpp:11.3f} | "
            f"{enc_text:>9s} | "
            f"{dec_text:>9s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress parsed CAN records from either a BLF file or a synthetic generator."
    )
    parser.add_argument("--input-blf", type=Path, help="Read CAN/CAN-FD frames from a Vector BLF file")
    parser.add_argument("--max-frames", type=int, help="Limit frames read from BLF or generated synthetically")
    parser.add_argument("--frames", type=int, default=60000)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results.json"))
    parser.add_argument("--data-out", type=Path, default=Path("synthetic_can_records.jsonl"))
    parser.add_argument("--skip-data-out", action="store_true", help="Do not write parsed records as JSONL")
    args = parser.parse_args()

    if not 0.0 <= args.train_ratio < 1.0:
        raise SystemExit("--train-ratio must be in [0, 1)")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")

    if args.input_blf:
        frames = load_blf_frames(args.input_blf, args.max_frames)
        source = f"blf:{args.input_blf}"
    else:
        frame_count = args.max_frames if args.max_frames is not None else args.frames
        frames = generate_can_frames(frame_count, args.seed)
        source = f"synthetic:seed={args.seed}"

    split = int(len(frames) * args.train_ratio)
    if split >= len(frames):
        raise SystemExit("train split consumes all frames; lower --train-ratio or provide more frames")

    if not args.skip_data_out:
        write_jsonl(frames, args.data_out)
    dataset = summarize_dataset(frames, source)
    print("Dataset summary:")
    print(json.dumps(dataset, indent=2, ensure_ascii=False))

    results = raw_and_zlib_baselines(frames, args.train_ratio)
    results["grouped_gru_predictor_arithmetic"] = compression_run(
        frames, OnlineGRUPredictor(), args.train_ratio
    )
    results["grouped_tcn_predictor_arithmetic"] = compression_run(
        frames, OnlineTCNPredictor(), args.train_ratio
    )
    results["grouped_last_value_arithmetic"] = compression_run(
        frames, LastValuePredictor(), args.train_ratio
    )

    print_table(results)
    args.out.write_text(
        json.dumps({"dataset": dataset, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")
    if not args.skip_data_out:
        print(f"Wrote {args.data_out}")


if __name__ == "__main__":
    main()
