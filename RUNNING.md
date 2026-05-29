# Running Guide

## Requirements

- Python 3.10 or newer
- No Python package installation is required for synthetic data
- BLF input requires `python-can`
- Optional external codecs:
  - `zstd` for Zstandard benchmarks
  - `xz` for LZMA2 / 7z-like benchmarks

The neural-style predictors are dependency-free online models implemented in
pure Python. They are intended for algorithm comparison and reproducible
experiments, not GPU training.

Install the optional BLF reader dependency only when you need BLF input:

```bash
python3 -m pip install -r requirements-blf.txt
```

## Quick Start

Run the default 60,000-frame experiment:

```bash
python3 can_compression_lab.py --frames 60000 --train-ratio 0.7
```

This writes:

- `synthetic_can_records.jsonl`: generated parsed CAN records
- `results.json`: benchmark metrics

## BLF Input

To compress from a real Vector BLF file, pass `--input-blf`. The script reads
CAN and CAN-FD messages with `python-can`, skips remote and error frames, and
converts every payload-bearing message into this parsed record shape:

```text
timestamp_delta_us, arbitration_id, dlc, payload_hex
```

Example:

```bash
python3 can_compression_lab.py \
  --input-blf /path/to/drive_log.blf \
  --train-ratio 0.7 \
  --out drive_log_results.json \
  --data-out drive_log_records.jsonl
```

For large BLF files, avoid writing the JSONL copy:

```bash
python3 can_compression_lab.py \
  --input-blf /path/to/drive_log.blf \
  --train-ratio 0.7 \
  --out drive_log_results.json \
  --skip-data-out
```

To test quickly on the first N payload frames:

```bash
python3 can_compression_lab.py \
  --input-blf /path/to/drive_log.blf \
  --max-frames 100000 \
  --out drive_log_100k_results.json \
  --skip-data-out
```

The reported `raw_bytes` are not BLF file bytes. They are bytes in the normalized
parsed-record stream used by the compressor:

```text
4 bytes timestamp_delta_us + 4 bytes arbitration_id + 1 byte dlc + payload bytes
```

This is intentional: the algorithm compresses CAN signals after parsing, not
the BLF container bytes.

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

You can also benchmark external codecs on a BLF-derived parsed-record stream:

```bash
python3 benchmark_codecs_5mb.py \
  --input-blf /path/to/drive_log.blf \
  --train-ratio 0.7 \
  --raw-out drive_log_eval.raw \
  --out drive_log_codec_results.json
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

- Add ASC/MDF readers that emit the same record shape as the BLF reader.
- Add a real range coder bitstream writer/reader behind the adaptive models.
- Compare per-signal residual coding after DBC decoding versus raw payload-byte
  residual coding.
