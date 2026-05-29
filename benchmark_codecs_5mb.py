#!/usr/bin/env python3
"""Benchmark external codecs on the same 5 MB synthetic CAN parsed records."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from can_compression_lab import generate_can_frames, serialize_frames


FRAMES = 250_000
TRAIN_RATIO = 0.7
SEED = 42


def run_checked(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def bench_codec(name: str, compress_cmd: list[str], decompress_cmd: list[str], raw_path: Path, out_path: Path) -> dict:
    t0 = time.perf_counter()
    run_checked(compress_cmd)
    enc_time = max(1e-9, time.perf_counter() - t0)

    t1 = time.perf_counter()
    decoded = subprocess.run(decompress_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    dec_time = max(1e-9, time.perf_counter() - t1)

    raw = raw_path.read_bytes()
    if decoded != raw:
        raise RuntimeError(f"{name} decode mismatch")

    raw_bytes = len(raw)
    compressed_bytes = out_path.stat().st_size
    raw_mb = raw_bytes / 1_000_000.0
    return {
        "frames": int(FRAMES * (1.0 - TRAIN_RATIO)),
        "raw_bytes": raw_bytes,
        "compressed_bytes_est": compressed_bytes,
        "compression_ratio_raw_over_compressed": raw_bytes / compressed_bytes,
        "encode_mb_per_sec": raw_mb / enc_time,
        "decode_mb_per_sec": raw_mb / dec_time,
    }


def main() -> None:
    frames = generate_can_frames(FRAMES, SEED)
    split = int(len(frames) * TRAIN_RATIO)
    raw = serialize_frames(frames[split:])
    raw_path = Path("synthetic_can_eval_5mb.raw")
    raw_path.write_bytes(raw)

    results = {}

    zst_path = Path("synthetic_can_eval_5mb.raw.zst")
    results["zstd_19_on_parsed_records"] = bench_codec(
        "zstd",
        ["zstd", "-19", "-f", str(raw_path), "-o", str(zst_path)],
        ["zstd", "-d", "-c", str(zst_path)],
        raw_path,
        zst_path,
    )

    xz_path = Path("synthetic_can_eval_5mb.raw.xz")
    results["xz_lzma2_9e_7zip_like"] = bench_codec(
        "xz",
        ["xz", "-9e", "-k", "-f", str(raw_path)],
        ["xz", "-d", "-c", str(xz_path)],
        raw_path,
        xz_path,
    )

    Path("codec_results_5mb.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
