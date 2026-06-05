# Dataset Card — UCC 5G Dataset (service-quality-checker usage)

---

## Dataset Summary

This project uses the **UCC 5G Dataset**, a publicly available collection of
drive-test measurements captured with the G-NetTrack Pro Android application
across 5G and LTE networks in Ireland. The dataset was used to train a binary
QoS Pass/Fail classifier that predicts whether a measurement point is likely to
deliver acceptable downlink throughput.

| Property | Value |
|---|---|
| Total rows | 188 711 |
| CSV files | 83 |
| Operator | Single anonymous Irish operator |
| Collection period | 2019–2020 (approximate) |
| App patterns | Amazon Prime Video, Netflix, File Download |
| Mobility conditions | Static, Driving |
| Label | `DL_bitrate ≥ 5 000 kbps` → Pass (1), else Fail (0) |
| Class balance | ~24.2 % Pass / ~75.8 % Fail |

---

## Source & Citation

**Repository:** https://github.com/uccmisl/5Gdataset

**Paper (mandatory citation):**

> Raca, D., Leahy, D., Sreenan, C. J., & Quinlan, J. J. (2020).
> *Beyond Throughput, the Next Generation: A 5G Dataset with Channel and
> Context Metrics.*
> Proceedings of the 11th ACM Multimedia Systems Conference (MMSys '20).
> ACM, New York, NY, USA.
> DOI / proceedings: https://dl.acm.org/doi/10.1145/3339825.3394938

The authors' requested form of attribution for use of this dataset is
**citing the above paper**.

---

## License

**License: GPL-3.0**

The UCC 5G dataset repository is released under the
[GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.html).

> **Important:** This dataset is licensed under GPL-3.0, **not** Creative
> Commons CC-BY. Any derivative work or software that incorporates or is
> trained on this data must comply with the terms of GPL-3.0, including
> the requirement to make source code of derivative works available under
> the same license if distributed.

This project's pipeline code is provided for research and educational purposes.
Users deploying derived models in production are responsible for assessing and
fulfilling their GPL-3.0 obligations.

---

## Composition

### Files
- 83 CSV files, each recording a single drive-test session
- Organized into subdirectories by application pattern and mobility condition
- Distributed as a single zip archive (`5G-production-dataset.zip`, ~2.1 MB compressed)

### Application patterns
| Pattern | Description |
|---|---|
| Amazon_Prime | Streaming video via Amazon Prime Video |
| Netflix | Streaming video via Netflix |
| Download | Active file download (maximum throughput) |

### Mobility conditions
| Condition | Description |
|---|---|
| Static | Device stationary (Speed ≈ 0 km/h) |
| Driving | Device in a moving vehicle |

### Geographic scope
Single anonymous Irish mobile operator. Data was collected in an unspecified
Irish urban/suburban area. No GPS coordinates are present in the published
dataset (anonymized).

### Class balance
Of 188 711 labeled rows: approximately **24.2 % Pass** (DL_bitrate ≥ 5 000 kbps)
and **75.8 % Fail**. Importantly, ~43.8 % of all rows have `DL_bitrate == 0`,
which correspond to streaming buffer pauses and idle periods rather than radio
failure (see Known Issues).

---

## Fields / Schema

The columns used by this project are listed below. The raw CSVs contain additional
columns (GPS, serving cell ID, NR-specific metrics) that are not used in the
current model.

| Column | Type | Unit | Description |
|---|---|---|---|
| `Timestamp` | string | — | Measurement time, format `YYYY.MM.DD_HH.MM.SS` |
| `NetworkMode` | string | — | Reported radio access technology (e.g., `LTE`, `NR`) |
| `RSRP` | float | dBm | Reference Signal Received Power. Valid: −140 to −40. Sentinel −200 scrubbed to NaN. |
| `RSRQ` | float | dB | Reference Signal Received Quality. Valid: −30 to +10. |
| `SNR` | float | dB | Signal-to-Noise Ratio. Valid: −30 to +40. Frequently missing. |
| `CQI` | float | 0–15 | Channel Quality Indicator. Valid: 0 to 15. Frequently missing. |
| `Speed` | float | km/h | Device speed at measurement. Valid: 0 to 130. |
| `DL_bitrate` | float | **kbps** | Downlink throughput. **Units are kbps, not Mbps.** Used to derive the label. |

> **Unit warning:** `DL_bitrate` is recorded in **kilobits per second (kbps)**.
> A value of 5 000 corresponds to 5 Mbps. Misreading this as Mbps would shift
> the pass threshold by a factor of 1 000.

The derived columns added by `src/sqc/features.py`:

| Column | Description |
|---|---|
| `ts` | Parsed datetime from `Timestamp` (Python `datetime`) |
| `label` | Binary label: `1` if `DL_bitrate ≥ 5 000`, else `0` |

---

## Label Derivation

```
label = 1  (Pass)   if  DL_bitrate ≥ 5 000 kbps  (≥ 5 Mbps)
label = 0  (Fail)   otherwise
```

The 5 000 kbps threshold is defined in `src/sqc/config.py` as
`PASS_THRESHOLD_KBPS = 5000` and can be overridden at feature-engineering time.
The threshold was chosen as a reasonable minimum for fluid video streaming.

**Class balance on the full 188 711-row dataset:**
- Pass (label = 1): ~45 738 rows (~24.2 %)
- Fail (label = 0): ~142 973 rows (~75.8 %)

The severe class imbalance is addressed in training with
`class_weight='balanced'` in the `RandomForestClassifier`, which up-weights the
minority Pass class during tree construction.

---

## Data Lineage

```
UCC GitHub repo (GPL-3.0)
        │
        │  scripts/get_data.py
        │  (downloads 5G-production-dataset.zip ~2.1 MB)
        ▼
data/raw/5G-production-dataset/   ← 83 CSV files (gitignored)
        │
        │  src/sqc/ingest.py   – locates all CSV files
        │  src/sqc/features.py – load_raw() → clean() → make_label()
        ▼
data/processed/dataset.parquet   ← 188 711 rows (gitignored)
        │
        │  src/sqc/train.py
        │  80/20 stratified split
        ▼
models/model.joblib               ← 60 MB RandomForest bundle (gitignored)
        │
        │  app/main.py (FastAPI)
        │  Docker image → Docker Hub
        ▼
https://musmanbinyounas-service-quality-checker.hf.space
```

Raw data, processed parquet, and model artifacts are all **gitignored** and
are never committed to the repository.

---

## Preprocessing & Cleaning

All preprocessing is implemented in `src/sqc/features.py` and
`src/sqc/validate.py`.

| Step | Detail |
|---|---|
| NA value tokens | G-NetTrack Pro writes `"-"` for missing fields; also empty strings. Both are read as `NaN` via `pd.read_csv(na_values=["-", ""])`. |
| RSRP sentinel | Values of exactly `−200.0` are a "no signal" placeholder. Scrubbed to `NaN` before training. |
| NRxRSRQ garbage | Values of `2 147 483 647` (INT_MAX) are a G-NetTrack Pro encoding artifact. Scrubbed to `NaN`. |
| Numeric coercion | All six numeric columns are coerced with `pd.to_numeric(errors="coerce")`. |
| Timestamp parsing | `Timestamp` string (format `YYYY.MM.DD_HH.MM.SS`) → Python `datetime` column `ts`. Unparseable entries become `NaT`. |
| Zero-throughput rows | **Kept** in the dataset and labeled as Fail. They represent application pauses, not radio failure. See Known Issues. |
| Pandera validation | `src/sqc/validate.py` enforces feature bounds (e.g., RSRP ∈ [−140, −40]) with nullable columns to allow residual NaN after sentinel scrubbing. |
| Missing-value imputation | `SimpleImputer(strategy="median")` is the first step in the sklearn `Pipeline`, applied at training time and baked into the saved model artifact. |

---

## Known Issues & Limitations

**1. Zero-throughput rows (43.8 % of data)**  
`DL_bitrate == 0` rows are labeled Fail by the threshold rule, but they represent
streaming buffer re-fill pauses and idle app states rather than genuine radio
degradation. This introduces systematic label noise in the Fail class.

**2. No cell-load or backhaul columns**  
The dataset does not contain columns describing cell occupancy, PRB utilization,
or backhaul congestion. Models trained on these data cannot distinguish coverage
failure from congestion-driven throughput loss — a fundamental limitation for
network operations triage.

**3. Single operator and single country**  
All measurements are from one anonymous Irish operator. The RSRP/RSRQ distributions,
cell planning assumptions, and frequency bands may differ from other operators or
countries. External validity is unvalidated.

**4. Temporal distribution shift**  
A PSI drift analysis (stage-8 CT evaluation) comparing the earlier 80 % of
rows to the later 20 % found Speed PSI ≈ 1.03 — significant drift in mobility
patterns over the collection period. Models trained on earlier data may be
mis-calibrated on later time periods.

**5. No subscriber or location identity**  
The published dataset is anonymized: no GPS coordinates, subscriber IDs, or
cell IDs are included. Applications requiring geographic localization of problems
must supplement with external mapping.

---

## Intended Use

This dataset card and the associated processing pipeline are provided for:
- **Research** into 5G/LTE QoS prediction and MLOps pipeline design
- **Education** — end-to-end demonstration of data ingestion, validation,
  model training, serving, monitoring, and continuous training
- **Baseline benchmarking** for radio-access QoS classifiers

Use of the dataset in any context must comply with the **GPL-3.0 license** and
must include the **paper citation** as requested by the authors.
