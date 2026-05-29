# Running Guide

## Requirements

- Python 3.10 or newer
- No Python package installation is required
- Optional external codecs:
  - `zstd` for Zstandard benchmarks
  - `xz` for LZMA2 / 7z-like benchmarks

The neural-style predictors are dependency-free online models implemented in
pure Python. They are intended for algorithm comparison and reproducible
experiments, not GPU training.

## Quick Start

Run the default 60,000-frame experiment:

```bash
python3 can_compression_lab.py --frames 60000 --train-ratio 0.7
```

This writes:

- `synthetic_can_records.jsonl`: generated parsed CAN records
- `results.json`: benchmark metrics

## 5 MB Experiment

Generate about 5 MB of parsed CAN records and run the same benchmark:

```bash
python3 can_compression_lab.py \
  --frames 250000 \
  --train-ratio 0.7 \
  --out results_5mb.json \
  --data-out synthetic_can_records_5mb.jsonl
```

The internal binary parsed-record stream is about 5.02 MB. The JSONL output is
larger because it is a human-readable text representation.

## External Codec Benchmark

After the 5 MB experiment, compare general-purpose codecs on the same
evaluation split:

```bash
python3 benchmark_codecs_5mb.py
```

The script benchmarks:

- `zstd -19`
- `xz -9e`, used here as a 7z-like LZMA2 baseline when the `7z` command is not
  installed

It writes:

- `codec_results_5mb.json`
- `synthetic_can_eval_5mb.raw`
- `synthetic_can_eval_5mb.raw.zst`
- `synthetic_can_eval_5mb.raw.xz`

The raw and compressed binary artifacts are ignored by Git.

## Methods Compared

- `zlib level 9 on parsed records`
- `zstd -19 on parsed records`
- `xz/LZMA2 -9e, 7z-like codec`
- grouped GRU-style predictor + adaptive entropy-code size estimate
- grouped TCN / causal 1D-CNN predictor + adaptive entropy-code size estimate
- grouped last-value predictor + adaptive entropy-code size estimate

The neural rows estimate arithmetic/range coding size with an adaptive
frequency model. They do not currently materialize a compressed bitstream.

## Interpreting Speed

Speed is reported as MB/s over the raw parsed-record evaluation stream:

```text
raw evaluation bytes / elapsed seconds / 1,000,000
```

For a 70/30 train/evaluation split, the 250,000-frame run evaluates on about
1.51 MB.

## Recreate Visuals

The report HTML files are static and can be opened directly:

- `results_report.html`
- `codec_comparison_5mb.html`

On macOS, a PNG thumbnail can be regenerated with:

```bash
qlmanage -t -s 1400 -o . codec_comparison_5mb.html
```

## Suggested Next Steps

- Replace the synthetic generator with a BLF/ASC/MDF parser that emits the same
  record shape: `timestamp_delta_us`, `arbitration_id`, `dlc`, `payload`.
- Add a real range coder bitstream writer/reader behind the adaptive models.
- Compare per-signal residual coding after DBC decoding versus raw payload-byte
  residual coding.
