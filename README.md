# Service Quality Checker

A binary 5G/LTE **QoS Pass/Fail classifier**. Given radio measurements at a
point (RSRP, RSRQ, SNR, CQI, speed), it predicts whether the location delivers
acceptable downlink service quality. It is a **screening / triage tool** that
flags points for human review — not a verdict engine (see `docs/MODEL_CARD.md`).

## Data

UCC 5G dataset — Raca, Leahy, Sreenan & Quinlan, *Beyond Throughput, The Next
Generation: A 5G Dataset with Channel and Context Metrics*, ACM MMSys 2020.
Repo: https://github.com/uccmisl/5Gdataset (repo license: GPL-3.0).

Raw data is **not** committed. Fetch it locally:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # installs deps from requirements.txt
python scripts/get_data.py  # downloads + extracts into data/raw/
```

Label: `Pass` if `DL_bitrate >= 5000` kbps (≥ 5 Mbps), configurable in
`src/sqc/config.py`. **Note:** `DL_bitrate` is in **kbps**, not Mbps.

## Project layout

```
src/sqc/      pipeline package (config, ingest, features, validate, train, drift, retrain)
app/          FastAPI service (/predict, /health)
tests/        pytest suite
scripts/      get_data.py CLI
docs/         model card + dataset card
data/         raw/ + processed/ (both gitignored)
```

## Status

- [x] 1. Source & version control
- [x] 2. Data ingestion & protection
- [ ] 3. Data validation
- [ ] 4. CI
- [ ] 5. Training
- [ ] 6. Deployment
- [ ] 7. Monitoring
- [ ] 8. Continuous training (simulated)
- [ ] 9. Governance
