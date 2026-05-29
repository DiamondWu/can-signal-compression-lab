# CAN Signal Compression Lab

This directory contains a self-contained CAN signal compression experiment.

Quick run:

```bash
python3 can_compression_lab.py --frames 60000 --train-ratio 0.7
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
installation is required.

For full setup, 5 MB benchmarking, external codec comparison, and report
generation instructions, see `RUNNING.md`.
