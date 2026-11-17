"""Central configuration: paths, sensor dictionary, units and analysis thresholds.

Every tunable number of the project lives here so that notebooks and the
pipeline use exactly the same values and sensitivity checks can vary them in
one place.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "ExampleChruchData_18-1-2024_29-11-2025.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "clean.parquet"
FEATURES_FILE = PROCESSED_DIR / "features.parquet"
RISK_FILE = PROCESSED_DIR / "risk.parquet"
CHANGE_LOG_FILE = PROCESSED_DIR / "preprocessing_log.csv"
REPORT_DIR = ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
RESULTS_JSON = TABLE_DIR / "results.json"

# --------------------------------------------------------------------------
# Raw file format
# --------------------------------------------------------------------------
CSV_SEP = ";"
CSV_DECIMAL = ","
MISSING_CODE = -255
TECHNICAL_COLUMNS = ["result", "table"]
TIME_COLUMN = "_time"
SOURCE_TZ = "UTC"
LOCAL_TZ = "Europe/Berlin"
FREQ = "20min"
READINGS_PER_HOUR = 3

# German measure name -> short English prefix, unit
MEASURES = {
    "Luft-Temperatur": ("temp", "degC", "air temperature"),
    "Luft-Feuchtigkeit": ("rh", "%", "relative humidity"),
    "Oberflächen-Temperatur": ("surf_temp", "degC", "surface temperature"),
    "Absolute-Luft-Feuchtigkeit": ("abs_hum", "g/m3", "absolute humidity"),
    "Taupunkt": ("dew_point", "degC", "dew point"),
}

# --------------------------------------------------------------------------
# Sensor dictionary. Client wording is the ONLY name used in client figures.
# --------------------------------------------------------------------------
SENSORS = {
    "AU": dict(client="Outside air", short="Outside", level="outside", has_surface=False),
    "O1": dict(client="Organ", short="Organ", level="upper", has_surface=True),
    "AWn": dict(client="North exterior wall (lower level)", short="North wall", level="lower", has_surface=True),
    "AWs": dict(client="South exterior wall (lower level)", short="South wall", level="lower", has_surface=True),
    "AWo": dict(client="East exterior wall (lower level)", short="East wall", level="lower", has_surface=True),
    "K1": dict(client="Art objects, north-west corner", short="Art objects NW", level="lower", has_surface=True),
    "K2": dict(client="Art objects, south wall", short="Art objects S", level="upper", has_surface=False),
    "D1": dict(client="Attic", short="Attic", level="attic", has_surface=False),
    "Ke1": dict(client="Basement storage area", short="Basement", level="basement", has_surface=False),
}
INDOOR = ["O1", "AWn", "AWs", "AWo", "K1", "K2", "D1", "Ke1"]
SURFACE_SENSORS = [s for s, v in SENSORS.items() if v["has_surface"]]


def client_name(code: str, short: bool = False) -> str:
    """Return the plain-language name of a sensor code for client output."""
    return SENSORS[code]["short" if short else "client"]


# --------------------------------------------------------------------------
# Preprocessing parameters
# --------------------------------------------------------------------------
# Physically plausible ranges (values outside become NaN and are logged)
PLAUSIBLE = {
    "temp": (-30.0, 50.0),
    "surf_temp": (-30.0, 50.0),
    "rh": (0.0, 110.0),          # >100 handled separately by clipping
    "abs_hum": (0.0, 40.0),
    "dew_point": (-40.0, 35.0),
}
INTERP_LIMIT = 3                 # max gap length (readings) that is interpolated = 1 h
WARMUP_READINGS = 6              # 2 h settling time after a sensor (re)start
RESTART_MIN_GAP = 18             # a gap >= 6 h (18 readings) counts as a restart
STUCK_MIN_RUN = 36               # >= 12 h identical temperature or humidity = stuck
SUN_AU_TEMP = 32.0               # outside readings above this in the evening are suspect
SUN_AU_HOURS = range(16, 22)     # local hours with direct evening sun on the west tower

# --------------------------------------------------------------------------
# Mold risk parameters
# --------------------------------------------------------------------------
# Approximate points read from Fig. 2 of the brief (Fraunhofer IBP, LIM_Bau)
LIM_POINTS = {
    "I": {0: 98.0, 5: 87.0, 10: 81.5, 15: 78.5, 20: 77.0, 25: 76.0, 30: 76.0},
    "II": {0: 100.0, 5: 90.5, 10: 85.0, 15: 82.0, 20: 80.0, 25: 79.0, 30: 79.0},
}
DEFAULT_LIM_CLASS = "I"
MARGIN_CRITICAL = 3.0            # dew point margin 0..3 degC = critical
MARGIN_SENSITIVITY = [2.0, 3.0, 4.0]
PERSIST_WINDOW = "7D"            # window for "humidity accumulating" rule
PERSIST_SHARE = 0.5              # >= 50 % of valid time above LIM in the window
MIN_VALID_SHARE = 0.5            # rolling windows need >= 50 % valid data
HEATING_MONTHS = [10, 11, 12, 1, 2, 3, 4]

# --------------------------------------------------------------------------
# Organ
# --------------------------------------------------------------------------
ORGAN_SENSOR = "O1"
ORGAN_DELTA_T = 1.0              # degC
ORGAN_WINDOW = "1h"

RANDOM_SEED = 42
