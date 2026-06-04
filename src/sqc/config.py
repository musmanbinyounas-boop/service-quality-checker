"""Central configuration for the Service Quality Checker.

Every value here was derived from inspecting all 188,711 rows of the UCC 5G
dataset (Raca et al., MMSys 2020), so the defaults match the *real* data, not
textbook assumptions. The two that matter most:

  * DL_bitrate is in **kbps**, not Mbps -> the Pass threshold is 5000, not 5.
  * Missing values are the string "-", not blanks -> see NA_VALUES.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (config.py lives at src/sqc/config.py, so repo root is two parents up) #
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# When the dataset zip is extracted it produces this top-level folder.
DATASET_DIRNAME = "5G-production-dataset"
EXTRACTED_DIR = RAW_DIR / DATASET_DIRNAME

PROCESSED_FILE = PROCESSED_DIR / "dataset.parquet"  # cleaned, labelled output

# --------------------------------------------------------------------------- #
# Data source                                                                 #
# --------------------------------------------------------------------------- #
# Single 2.1 MB zip living inside the GitHub repo (not 166 loose files).
DATA_URL = "https://github.com/uccmisl/5Gdataset/raw/master/5G-production-dataset.zip"
ZIP_NAME = "5G-production-dataset.zip"
# Fallback if the raw download is blocked on your network:
#   git clone --depth 1 https://github.com/uccmisl/5Gdataset.git
GIT_FALLBACK = "https://github.com/uccmisl/5Gdataset.git"

# The zip ships a macOS junk folder we must never read as data.
JUNK_DIR_MARKER = "__MACOSX"

# --------------------------------------------------------------------------- #
# Label definition                                                            #
# --------------------------------------------------------------------------- #
THROUGHPUT_COL = "DL_bitrate"      # downlink throughput, units: kbps
PASS_THRESHOLD_KBPS = 5000         # >= 5 Mbps == Pass; override via env/CLI
LABEL_COL = "label"                # 1 = Pass, 0 = Fail

# --------------------------------------------------------------------------- #
# Features                                                                    #
# --------------------------------------------------------------------------- #
FEATURES = ["RSRP", "RSRQ", "SNR", "CQI", "Speed"]

# Missing-value tokens in the raw CSVs (G-NetTrack Pro writes "-").
NA_VALUES = ["-", ""]

# Sentinels to scrub to NaN before validation/training.
RSRP_SENTINEL = -200               # "no signal" placeholder
NRX_RSRQ_GARBAGE = 2147483647      # INT_MAX garbage seen in NRxRSRQ

# --------------------------------------------------------------------------- #
# Empirical validation bounds (inclusive). Widened from the textbook ranges   #
# so that stage-3 validation does NOT reject genuine rows (e.g. ~8% of real   #
# RSRQ falls outside the textbook -19.5..-3 band).                            #
# --------------------------------------------------------------------------- #
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "RSRP": (-140.0, -40.0),
    "RSRQ": (-30.0, 10.0),
    "SNR": (-30.0, 40.0),
    "CQI": (0.0, 15.0),
    "Speed": (0.0, 130.0),
}

# Raw column dtypes we expect after reading with NA_VALUES (for sanity checks).
RAW_NUMERIC_COLS = ["RSRP", "RSRQ", "SNR", "CQI", "Speed", THROUGHPUT_COL]

# Reproducibility
RANDOM_STATE = 42
