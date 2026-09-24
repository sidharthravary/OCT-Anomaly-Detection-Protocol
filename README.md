# OCT Anomaly Detection Protocol

One-class anomaly detection on retinal OCT B-scans (NEH dataset: Normal / Drusen / CNV).
Models are trained only on normal scans and flag Drusen and CNV as anomalies.
See [PRD.md](PRD.md) for the full protocol.

## Setup
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```
Set `data_root` in `config.yaml`, then run the scripts in order (`01_...` to `11_...`).
