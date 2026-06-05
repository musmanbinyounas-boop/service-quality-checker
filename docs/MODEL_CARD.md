# Model Card — Service Quality Checker

**Version:** 1.0.0 · **Last trained:** 2026-06-04 · **Maintainer:** M. Usman Bin Younas

---

## Model Overview

| Field | Value |
|---|---|
| Model type | scikit-learn `RandomForestClassifier` (Pipeline) |
| Task | Binary classification — Pass / Fail QoS screening |
| Input | 5 radio/mobility measurements (see Features) |
| Output | `Pass` (1) or `Fail` (0) + pass probability |
| Threshold | DL_bitrate ≥ 5 000 kbps → Pass |
| Trained at | 2026-06-04T20:16:57 UTC |
| sklearn version | 1.9.0 |
| Training set size | 150 968 rows |
| Random state | 42 |

This model is a **screening and triage tool**. It flags measurement points as
potentially poor-quality for human review. It is not a definitive verdict engine
and must not be used for automated penalties, billing adjustments, or SLA
enforcement without human oversight.

---

## Intended Use

### In-scope
- Identifying geographic or temporal clusters of degraded 5G/LTE service for
  network-operations triage
- Exploratory QoS analysis on datasets with the same five radio features
- Research and teaching demonstrations of MLOps end-to-end pipelines

### Out-of-scope / prohibited
- **Automated penalties or SLA breach rulings** — model precision on the Pass
  class is ~0.53; roughly 1 in 2 "Pass" predictions is a false alarm at this
  threshold. Human review is required before any consequential decision.
- **Billing or contractual enforcement** against operators or subscribers
- **Cross-operator production deployment** without revalidation — the model was
  trained on a single anonymous Irish operator's data (see Limitations §3)
- **Real-time closed-loop network control** — outputs are probabilities, not
  verified measurements

---

## Training Data

Training data is derived from the **UCC 5G Dataset** (Raca et al., MMSys 2020).
See [`docs/DATASET_CARD.md`](DATASET_CARD.md) for the full lineage, schema,
preprocessing steps, and known issues.

**Summary:** 188 711 rows from 83 drive-test CSV files, single Irish operator,
three application patterns (Amazon Prime, Netflix, Download) × two mobility
conditions (Static, Driving). Label: `DL_bitrate ≥ 5 000 kbps` → Pass.
Class balance: ~24.2 % Pass / 75.8 % Fail (imbalanced; addressed with
`class_weight='balanced'` in the classifier).

80 % of rows (150 968) were used for training; the remaining 20 % (37 743) formed
a held-out test set used for all reported metrics below.

---

## Features & Label

| Feature | Description | Unit | Valid range | Missing handling |
|---|---|---|---|---|
| RSRP | Reference Signal Received Power | dBm | −140 to −40 | Imputed (median) |
| RSRQ | Reference Signal Received Quality | dB | −30 to +10 | Imputed (median) |
| SNR | Signal-to-Noise Ratio | dB | −30 to +40 | Imputed (median) |
| CQI | Channel Quality Indicator | 0–15 (integer) | 0 to 15 | Imputed (median) |
| Speed | Device speed at measurement time | km/h | 0 to 130 | Imputed (median) |

**Label:** `label = 1` (Pass) if `DL_bitrate ≥ 5 000 kbps`, else `label = 0`
(Fail). `DL_bitrate` is downlink throughput in **kbps** (not Mbps).

Missing values arise from G-NetTrack Pro writing `"-"` as a sentinel and from
genuine measurement gaps (SNR and CQI are absent when the device is not attached
to 5G NR). All five features are passed through a `SimpleImputer(strategy="median")`
before the classifier.

---

## Model Architecture

```
PredictRequest (5 floats)
        │
        ▼
SimpleImputer(strategy="median")   ← handles NaN / missing features
        │
        ▼
RandomForestClassifier(
    n_estimators   = 200,
    min_samples_leaf = 5,
    class_weight   = "balanced",   ← compensates 76 % Fail majority
    random_state   = 42,
    n_jobs         = -1
)
        │
        ▼
  label ∈ {0, 1}  +  pass_probability ∈ [0, 1]
```

A `LogisticRegression` pipeline (SimpleImputer → StandardScaler → LR) was also
trained and evaluated. `RandomForest` was selected because it achieved a higher
F1 score on the Pass class (0.622 vs 0.416) on the held-out test set. See the
Evaluation section for the full comparison.

---

## Evaluation

Metrics are reported for the **Pass class (label = 1)**, which is the
operationally relevant class (identifying locations with adequate throughput).

**Test set:** 37 743 rows — 9 132 Pass (24.2 %), 28 611 Fail (75.8 %)  
**Split:** stratified random 80/20, `random_state=42`

| Model | Precision (Pass) | Recall (Pass) | F1 (Pass) |
|---|:---:|:---:|:---:|
| LogisticRegression | 0.306 | 0.647 | 0.416 |
| **RandomForest** ✓ | **0.527** | **0.760** | **0.622** |

**Why RandomForest was chosen:** higher F1 on the minority Pass class (+0.206),
substantially better precision (+0.221) without sacrificing recall, and higher
overall accuracy (78 % vs 56 %). The balanced class-weight setting means both
models are tuned to maximize recall on the Pass class rather than raw accuracy.

**Confusion matrix (RandomForest, test set):**

|  | Predicted Fail | Predicted Pass |
|---|:---:|:---:|
| **Actual Fail** (28 611) | 22 383 (TN) | 6 228 (FP) |
| **Actual Pass** (9 132) | 2 194 (FN) | 6 938 (TP) |

The model misses 2 194 genuinely passing points (false negatives) and raises
6 228 false alarms (false positives). This asymmetry is intentional: missing a
degraded location is worse than reviewing a healthy one, so the model is calibrated
toward recall.

---

## Monitoring & Continuous Training

**Live monitoring:** every `/predict` call on the deployed Space logs a document to
MongoDB Atlas (`sqc_monitoring.predictions`) containing the input features, label,
pass probability, model type, and latency. A background task handles the write
asynchronously so the database never adds latency to the inference path, and a
failed write never affects the prediction response.

**Drift detection:** `src/sqc/drift.py` computes Population Stability Index (PSI)
per feature against the committed training reference distribution
(`reports/train_feature_stats.json`). An alert is raised when `max_PSI ≥ 0.20`.

**Continuous training:** `src/sqc/retrain.py` implements a simulated CT pipeline
(see important note below). A GitHub Actions cron workflow (`.github/workflows/ct.yml`)
runs weekly and uploads the CT report and candidate model as Actions artifacts.
**The cron job does not auto-deploy to the Space.** Promotion requires manual
`--promote` after human review of the CT report.

> **Simulation note:** production `/predict` traffic is unlabeled — the tool
> receives radio measurements but not ground-truth throughput. Continuous training
> therefore uses a temporally held-back slice of the static UCC dataset as a proxy
> for "new arrivals." A CT run in June 2026 measured Speed PSI ≈ 1.03 (significant
> drift) between the training and holdback slices, and the promotion gate recommended
> KEEP_CURRENT (retrained F1 0.493 < baseline F1 0.540 on the fixed evaluation set).

---

## Limitations

These limitations should be understood before any operational use.

**1. Congestion vs. coverage ambiguity**  
The dataset contains no cell-load, capacity, or backhaul columns. A `Fail`
prediction cannot distinguish genuine poor radio coverage from temporary network
congestion. This is the primary reason the tool is framed as a *triage* instrument
that surfaces candidate problem points for field engineers to investigate, not a
system that renders a verdict.

**2. Label noise from zero-throughput rows**  
Approximately 43.8 % of rows have `DL_bitrate == 0`, corresponding to application
streaming buffer pauses or idle periods rather than radio failure. These rows
receive `label = Fail`, introducing noise: some "Fail" predictions reflect
application-layer behavior, not the radio channel.

**3. Single-operator, single-country scope**  
All data originates from one anonymous Irish operator. Cross-operator and
cross-region generalization has not been validated and must be treated as future
work. Feature distributions (especially RSRP and RSRQ) can vary significantly
across network vendors and regulatory bands.

**4. Temporal distribution shift is real and measured**  
The stage-8 CT evaluation found Speed PSI ≈ 1.03 between the early-period training
slice and the later holdback slice — a level classified as "high drift". Mobility
patterns change over calendar time (seasons, events, infrastructure upgrades).
Periodic revalidation against recent labeled data is necessary to maintain
calibration.

**5. Modest precision on the Pass class**  
Precision on `Pass` is ~0.53, meaning roughly half of locations predicted as
passing are false alarms. The model is deliberately tuned to favor recall (~0.76)
so that few genuinely degraded locations are missed at the cost of more false
positives entering the human-review queue. Operators prioritizing review queue
efficiency may need to raise the classification threshold or collect more balanced
labeled data.

---

## Ethical & Regulatory Considerations

- **Human in the loop:** all `Fail` predictions are intended for human review.
  No automated action (penalty, rerouting, subscriber notification) should be
  triggered from model outputs alone.
- **No demographic data:** inputs are radio measurements and device speed only;
  the model does not process subscriber identity, location coordinates, or any
  personal data.
- **Operator anonymity:** the training dataset uses data from an anonymized
  operator; no operator-identifiable information is present in the model or its
  outputs.
- **Regulatory note:** deployment in contexts subject to telecommunications
  regulation (e.g., QoS reporting obligations) requires review by qualified
  legal and engineering personnel. This tool is a research prototype.

---

## Reproducibility

| Parameter | Value |
|---|---|
| `RANDOM_STATE` | 42 |
| Train/test split | 80 / 20, stratified on label |
| sklearn version | 1.9.0 |
| Python | 3.11 |
| Imputer strategy | median |
| RF `n_estimators` | 200 |
| RF `min_samples_leaf` | 5 |
| RF `class_weight` | balanced |

All hyperparameters are defined in `src/sqc/config.py` and `src/sqc/train.py`.
The dataset download, feature engineering, training, and serving steps are
fully automated and documented in the project README.

---

## Citation & Attribution

If you use this model or the pipeline code, please also cite the underlying dataset:

> Raca, D., Leahy, D., Sreenan, C. J., & Quinlan, J. J. (2020).
> *Beyond Throughput, the Next Generation: A 5G Dataset with Channel and Context Metrics.*
> Proceedings of the 11th ACM Multimedia Systems Conference (MMSys '20).
> https://github.com/uccmisl/5Gdataset
