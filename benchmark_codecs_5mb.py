#!/usr/bin/env python3
"""Benchmark external codecs on the same 5 MB synthetic CAN parsed records."""

from __future__ import annotations

import json
import subprocess
import time
import argparse
from pathlib import Path

from can_compression_lab import generate_can_frames, load_blf_frames, serialize_frames


def run_checked(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def bench_codec(
    name: str,
    compress_cmd: list[str],
    decompress_cmd: list[str],
    raw_path: Path,
    out_path: Path,
    eval_frames: int,
) -> dict:
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
        "frames": eval_frames,
        "raw_bytes": raw_bytes,
        "compressed_bytes_est": compressed_bytes,
        "compression_ratio_raw_over_compressed": raw_bytes / compressed_bytes,
        "encode_mb_per_sec": raw_mb / enc_time,
        "decode_mb_per_sec": raw_mb / dec_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark external codecs on parsed CAN records.")
    parser.add_argument("--input-blf", type=Path, help="Read CAN/CAN-FD frames from a Vector BLF file")
    parser.add_argument("--frames", type=int, default=250_000)
    parser.add_argument("--max-frames", type=int, help="Limit frames read from BLF or generated synthetically")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-out", type=Path, default=Path("synthetic_can_eval_5mb.raw"))
    parser.add_argument("--out", type=Path, default=Path("codec_results_5mb.json"))
    args = parser.parse_args()

    if not 0.0 <= args.train_ratio < 1.0:
        raise SystemExit("--train-ratio must be in [0, 1)")

    if args.input_blf:
        frames = load_blf_frames(args.input_blf, args.max_frames)
    else:
        frame_count = args.max_frames if args.max_frames is not None else args.frames
        frames = generate_can_frames(frame_count, args.seed)

    split = int(len(frames) * args.train_ratio)
    if split >= len(frames):
        raise SystemExit("train split consumes all frames; lower --train-ratio or provide more frames")

    raw = serialize_frames(frames[split:])
    raw_path = args.raw_out
    raw_path.write_bytes(raw)
    eval_frames = len(frames) - split

    results = {}

    zst_path = raw_path.with_suffix(raw_path.suffix + ".zst")
    results["zstd_19_on_parsed_records"] = bench_codec(
        "zstd",
        ["zstd", "-19", "-f", str(raw_path), "-o", str(zst_path)],
        ["zstd", "-d", "-c", str(zst_path)],
        raw_path,
        zst_path,
        eval_frames,
    )

    xz_path = raw_path.with_suffix(raw_path.suffix + ".xz")
    results["xz_lzma2_9e_7zip_like"] = bench_codec(
        "xz",
        ["xz", "-9e", "-k", "-f", str(raw_path)],
        ["xz", "-d", "-c", str(xz_path)],
        raw_path,
        xz_path,
        eval_frames,
    )

    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
