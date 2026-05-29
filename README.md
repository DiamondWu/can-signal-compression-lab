# CAN Signal Compression Lab

This directory contains a self-contained CAN signal compression experiment.
It can run on either synthetic CAN/CAN-FD traffic or a real Vector BLF file.

Quick run:

```bash
python3 can_compression_lab.py --frames 60000 --train-ratio 0.7
```

Run on a BLF file:

```bash
python3 -m pip install -r requirements-blf.txt
python3 can_compression_lab.py \
  --input-blf path/to/input.blf \
  --train-ratio 0.7 \
  --out blf_results.json \
  --data-out blf_records.jsonl
```

The script generates realistic simulated CAN frames, groups them by arbitration
ID, predicts payload bytes per ID, and estimates arithmetic/range coding size
from residual probability models. It reports encode/decode throughput and
compression ratio for:

- grouped neural residual predictor (`gru`)
- grouped causal TCN/1D-CNN residual predictor (`tcn`)
- baselines (`raw`, `zlib`)

Outputs:

- `synthetic_can_records.jsonl`: simulated parsed CAN/CAN-FD records
- `results.json`: dataset summary and benchmark metrics

The implementation uses only the Python standard library. No dependency
installation is required for synthetic data. BLF input requires the optional
`python-can` dependency listed in `requirements-blf.txt`.

For full setup, 5 MB benchmarking, external codec comparison, and report
generation instructions, see `RUNNING.md`.
