"""Load, clean and save the church sensor data.

Run the whole preprocessing with a single command from the repository root::

    python -m src.load_data

Steps (each one is logged in ``data/processed/preprocessing_log.csv``):

1.  read the semicolon separated file with decimal comma
2.  drop the technical columns ``result`` and ``table``
3.  parse ``_time`` as UTC, check for duplicates and the 20-minute grid
4.  rename columns to short English names (``temp_O1``, ``rh_AWn`` ...)
5.  replace the missing code ``-255`` with NaN
6.  trim leading / trailing rows without any measurement
7.  commissioning period and documented manual exclusions -> NaN / removed
7b. plausibility ranges (out of range -> NaN)
8.  physical fixes: relative humidity > 100 % clipped, dew point <= air
    temperature, absolute humidity recomputed (flagged)
8b. isolated joint temperature + humidity glitches -> NaN (flagged)
9.  sensor warm-up after a restart -> NaN (flagged)
10. stuck-sensor check (long runs of identical values -> NaN)
11. flag suspected direct sun on the outside sensor (values kept)
12. convert the index to Europe/Berlin
13. interpolate gaps of at most ``INTERP_LIMIT`` readings (flagged)
14. final assertions and save to parquet

No value is changed silently: every change creates a log entry and, where it
matters for later analysis, a boolean flag column.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .physics import absolute_humidity, dew_point

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Change log
# --------------------------------------------------------------------------
@dataclass
class ChangeLog:
    """Collects every modification made to the data."""

    rows: list = field(default_factory=list)

    def add(self, step: str, column: str, n: int, action: str, reason: str) -> None:
        self.rows.append(dict(step=step, column=column, n_values=int(n), action=action, reason=reason))
        if n:
            log.info("%-22s %-28s %7d  %s", step, column, n, action)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["step", "column", "n_values", "action", "reason"])


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def sensor_of(col: str) -> str:
    """Sensor code of a cleaned column name, e.g. ``surf_temp_K1`` -> ``K1``."""
    return col.rsplit("_", 1)[1]


def measure_of(col: str) -> str:
    """Measure prefix of a cleaned column name, e.g. ``surf_temp_K1`` -> ``surf_temp``."""
    return col.rsplit("_", 1)[0]


def measurement_columns(df: pd.DataFrame) -> list[str]:
    """All sensor measurement columns (no flags)."""
    prefixes = {v[0] for v in C.MEASURES.values()}
    return [c for c in df.columns if "_" in c and measure_of(c) in prefixes]


def sensor_columns(df: pd.DataFrame, sensor: str) -> list[str]:
    return [c for c in measurement_columns(df) if sensor_of(c) == sensor]


def run_lengths(mask: pd.Series) -> pd.DataFrame:
    """Start, end and length of every run of True values in a boolean series."""
    m = mask.to_numpy(dtype=bool)
    if not m.any():
        return pd.DataFrame(columns=["start", "end", "length"])
    d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return pd.DataFrame(
        {"start": mask.index[starts], "end": mask.index[ends - 1], "length": ends - starts,
         "i_start": starts, "i_end": ends}
    )


def gap_lengths(s: pd.Series) -> pd.Series:
    """Lengths (in readings) of all interior NaN gaps of a series."""
    s = s.loc[s.first_valid_index(): s.last_valid_index()]
    return run_lengths(s.isna())["length"].astype(int)


# --------------------------------------------------------------------------
# Pipeline steps
# --------------------------------------------------------------------------
def read_raw(path=C.RAW_FILE) -> pd.DataFrame:
    """Read the raw CSV exactly as delivered (decimal comma, ``;`` separator)."""
    df = pd.read_csv(path, sep=C.CSV_SEP, decimal=C.CSV_DECIMAL, encoding="utf-8")
    return df


def rename_columns(columns) -> dict:
    """Map ``<Measure>_<Sensor>`` German names to ``<measure>_<sensor>``."""
    mapping = {}
    for c in columns:
        if "_" not in c or c.startswith("_"):
            continue
        measure, sensor = c.rsplit("_", 1)
        if measure in C.MEASURES:
            mapping[c] = f"{C.MEASURES[measure][0]}_{sensor}"
    return mapping


def basic_parse(raw: pd.DataFrame, clog: ChangeLog) -> pd.DataFrame:
    """Steps 2-6: drop technical columns, time index, rename, missing code, grid, trim."""
    df = raw.drop(columns=[c for c in C.TECHNICAL_COLUMNS if c in raw.columns])
    clog.add("drop_columns", ",".join(C.TECHNICAL_COLUMNS), len(raw), "dropped", "technical InfluxDB export columns")

    t = pd.to_datetime(df.pop(C.TIME_COLUMN), utc=True, format="ISO8601")
    df.index = pd.DatetimeIndex(t, name="time")

    n_dup = int(df.index.duplicated().sum())
    if n_dup:
        df = df[~df.index.duplicated(keep="first")]
    clog.add("duplicates", "index", n_dup, "dropped (keep first)", "duplicate timestamps")
    df = df.sort_index()

    df = df.rename(columns=rename_columns(df.columns))
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:  # should never happen with decimal=","
        for c in non_numeric:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        clog.add("dtype", ",".join(non_numeric), len(non_numeric), "coerced to float", "non numeric column")
    df = df.astype("float64")

    for c in df.columns:
        n = int((df[c] == C.MISSING_CODE).sum())
        df.loc[df[c] == C.MISSING_CODE, c] = np.nan
        clog.add("missing_code", c, n, "-255 -> NaN", "logger missing value code")

    full = pd.date_range(df.index.min(), df.index.max(), freq=C.FREQ, tz="UTC", name="time")
    n_new = len(full) - len(df)
    off_grid = int((~df.index.isin(full)).sum())
    df = df.reindex(full)
    clog.add("grid", "index", n_new, "rows inserted as NaN", "complete 20-minute grid")
    clog.add("grid", "index", off_grid, "rows off the 20-minute grid dropped", "irregular timestamps")

    has_data = df.notna().any(axis=1)
    first, last = has_data.idxmax(), has_data[::-1].idxmax()
    n_trim = int((df.index < first).sum() + (df.index > last).sum())
    df = df.loc[first:last]
    clog.add("trim", "index", n_trim, "leading/trailing empty rows removed", "no sensor reported")
    return df


def commissioning(df: pd.DataFrame, clog: ChangeLog) -> pd.DataFrame:
    """Step 7: remove the installation period and apply documented manual exclusions.

    Until 13:40 local time on 19 Jan 2024 every sensor reads 19-24 degC at the
    same time (installation bench), so these rows do not describe the church.
    Manual exclusions are listed with their reason in ``config.MANUAL_EXCLUSIONS``.
    """
    cut = pd.Timestamp(C.COMMISSIONING_END, tz=C.LOCAL_TZ).tz_convert("UTC")
    n = int((df.index < cut).sum())
    n_vals = int(df.loc[df.index < cut].notna().sum().sum())
    df = df.loc[df.index >= cut].copy()
    clog.add("commissioning", "all", n_vals, f"{n} rows before {C.COMMISSIONING_END} local removed",
             "sensors still at the installation bench")
    for ex in C.MANUAL_EXCLUSIONS:
        a = pd.Timestamp(ex["start"], tz=C.LOCAL_TZ).tz_convert("UTC")
        b = pd.Timestamp(ex["end"], tz=C.LOCAL_TZ).tz_convert("UTC")
        cols = [f"{m}_{ex['sensor']}" for m in ex["measures"]]
        rows = (df.index >= a) & (df.index <= b)
        n_vals = int(df.loc[rows, cols].notna().sum().sum())
        df.loc[rows, cols] = np.nan
        df[f"flag_manual_{ex['sensor']}"] = rows
        clog.add("manual_exclusion", ",".join(cols), n_vals, f"{ex['start']} - {ex['end']} -> NaN", ex["reason"])
    return df


def spikes(df: pd.DataFrame, clog: ChangeLog) -> pd.DataFrame:
    """Step 8b: remove isolated glitches where temperature and humidity jump together.

    A reading is a glitch if its air temperature deviates by more than
    ``SPIKE_TEMP`` and its relative humidity by more than ``SPIKE_RH`` from the
    mean of both neighbours, both neighbours agree with each other, and the
    signal returns at the next reading. A real 1-2 degC drop of the room air for
    20 minutes in a massive church at night, without any response of the
    surface sensor, is not physical. Some glitch values are suspiciously round
    (exactly 11.00 or 12.00 degC). Each such glitch would create two false
    organ events.
    """
    def is_spike(x: pd.Series, thr: float) -> pd.Series:
        prev, nxt = x.shift(1), x.shift(-1)
        dev = x - (prev + nxt) / 2
        return (dev.abs() > thr) & ((nxt - prev).abs() < dev.abs() / 2) & (np.sign(x - prev) != np.sign(nxt - x))

    for s in C.SENSORS:
        bad = is_spike(df[f"temp_{s}"], C.SPIKE_TEMP) & is_spike(df[f"rh_{s}"], C.SPIKE_RH)
        cols = [f"{m}_{s}" for m in ("temp", "rh", "abs_hum", "dew_point")]
        df.loc[bad, cols] = np.nan
        df[f"flag_spike_{s}"] = bad.to_numpy()
        clog.add("spike", f"temp_{s},rh_{s}", bad.sum(), "joint single-reading glitch -> NaN",
                 "isolated sensor glitch (temperature and humidity jump and return)")
    return df


def plausibility(df: pd.DataFrame, clog: ChangeLog) -> pd.DataFrame:
    """Step 7b: set values outside physically plausible ranges to NaN."""
    for c in measurement_columns(df):
        lo, hi = C.PLAUSIBLE[measure_of(c)]
        bad = (df[c] < lo) | (df[c] > hi)
        df.loc[bad, c] = np.nan
        clog.add("plausibility", c, bad.sum(), f"outside [{lo}, {hi}] -> NaN", "physically implausible")
    return df


def physical_fixes(df: pd.DataFrame, clog: ChangeLog) -> pd.DataFrame:
    """Step 8: relative humidity > 100 % and dew point above air temperature.

    Capacitive humidity sensors saturate and over-read in fog / rain. Air
    cannot hold more than 100 % relative humidity, so values are clipped to
    100 %, the dew point is set to the air temperature and the absolute
    humidity recomputed at saturation. A flag keeps the information.
    """
    for s in C.SENSORS:
        rh, t, dp, ah = f"rh_{s}", f"temp_{s}", f"dew_point_{s}", f"abs_hum_{s}"
        if rh not in df:
            continue
        over = df[rh] > 100
        under = df[rh] < 0
        dp_over = df[dp] > df[t]
        flag = over | dp_over
        df.loc[over, rh] = 100.0
        df.loc[under, rh] = 0.0
        df.loc[dp_over, dp] = df.loc[dp_over, t]
        df.loc[over, ah] = absolute_humidity(df.loc[over, t], 100.0)
        clog.add("physical_fix", rh, over.sum(), "clipped to 100 %", "sensor saturation")
        clog.add("physical_fix", rh, under.sum(), "clipped to 0 %", "negative humidity")
        clog.add("physical_fix", dp, dp_over.sum(), "set to air temperature", "dew point above air temperature")
        clog.add("physical_fix", ah, over.sum(), "recomputed at 100 %", "consistency with clipped humidity")
        if flag.any():
            df[f"flag_rh_clipped_{s}"] = flag
    return df


def warmup_artifacts(df: pd.DataFrame, clog: ChangeLog) -> pd.DataFrame:
    """Step 9: remove the settling period after a sensor restart.

    After long outages the sensors first report a value far from the room and
    then relax towards the room value within about an hour (for example the
    basement on 26 Jul 2024: 23.5 -> 18.6 degC, the east wall on 5 Sep 2025:
    24.2 -> 19.7 degC). These readings do not describe the church and would
    create false rapid-change events. (The installation day itself is handled
    by the commissioning step.)
    """
    for s in C.SENSORS:
        cols = sensor_columns(df, s)
        key = f"temp_{s}"
        valid = df[key].notna()
        gaps = run_lengths(~valid)
        restart_idx = []
        for _, g in gaps.iterrows():
            if g["length"] >= C.RESTART_MIN_GAP and g["i_end"] < len(df):
                restart_idx.append(int(g["i_end"]))
        flag = np.zeros(len(df), dtype=bool)
        for i in sorted(set(restart_idx)):
            flag[i: i + C.WARMUP_READINGS] = True
        flag &= valid.to_numpy()
        df.loc[flag, cols] = np.nan
        df[f"flag_warmup_{s}"] = flag
        clog.add("warmup", key, flag.sum(),
                 f"first {C.WARMUP_READINGS} readings after {len(set(restart_idx))} (re)starts -> NaN",
                 "sensor settling after installation / outage")
    return df


def stuck_values(df: pd.DataFrame, clog: ChangeLog) -> tuple[pd.DataFrame, pd.Series]:
    """Step 10: detect runs of identical values (a frozen sensor).

    Runs at exactly 100 % relative humidity are excluded: they are the
    clipped saturation readings of step 8, not a frozen sensor.
    Only temperature and relative humidity are checked (derived quantities
    can legitimately be constant at 0.01 resolution). Runs of at least
    ``STUCK_MIN_RUN`` readings are set to NaN except the first value.
    Returns the frame and the longest run per column for the report.
    """
    longest = {}
    for c in measurement_columns(df):
        x = df[c]
        same = x.diff().eq(0) & x.notna()
        if measure_of(c) == "rh":
            same &= x.lt(100.0)  # runs at 100 % are sensor saturation (clipped), not a frozen sensor
        runs = run_lengths(same)
        longest[c] = int(runs["length"].max() + 1) if len(runs) else 1
        if measure_of(c) not in ("temp", "rh", "surf_temp"):
            continue
        bad = np.zeros(len(df), dtype=bool)
        for _, r in runs[runs["length"] + 1 >= C.STUCK_MIN_RUN].iterrows():
            bad[r["i_start"]: r["i_end"]] = True
        df.loc[bad, c] = np.nan
        clog.add("stuck", c, bad.sum(), f">= {C.STUCK_MIN_RUN} identical readings -> NaN", "frozen sensor")
    return df, pd.Series(longest, name="longest_identical_run")


def sun_flag(df: pd.DataFrame, clog: ChangeLog) -> pd.DataFrame:
    """Step 11: flag suspected direct evening sun on the outside sensor.

    The outside sensor is mounted at the west tower. Its daily maximum occurs
    around 19-20 h local time (normal: 14-16 h) and reaches 40 degC, which is
    above any weather-station value for the region. Values are *kept* (they
    are a real local heat load on the facade) but flagged so that analyses of
    the general weather can exclude them.
    """
    hour = df.index.tz_convert(C.LOCAL_TZ).hour
    flag = (df["temp_AU"] > C.SUN_AU_TEMP) & np.isin(hour, list(C.SUN_AU_HOURS))
    df["flag_sun_AU"] = flag
    clog.add("sun_flag", "temp_AU", flag.sum(), "flagged (kept)", "suspected direct sun on the sensor")
    return df


def interpolate_short_gaps(s: pd.Series, limit: int = C.INTERP_LIMIT) -> tuple[pd.Series, pd.Series]:
    """Linearly interpolate interior gaps whose *total* length is <= ``limit``.

    ``pandas.interpolate(limit=...)`` would partially fill longer gaps; here a
    gap is either filled completely or not at all. Returns the filled series
    and a boolean mask of filled positions.
    """
    na = s.isna()
    runs = run_lengths(na)
    fill = np.zeros(len(s), dtype=bool)
    for _, r in runs.iterrows():
        interior = r["i_start"] > 0 and r["i_end"] < len(s)
        if interior and r["length"] <= limit:
            fill[r["i_start"]: r["i_end"]] = True
    filled = s.interpolate(method="time", limit_area="inside")
    out = s.copy()
    out[fill] = filled[fill]
    return out, pd.Series(fill, index=s.index)


def interpolate(df: pd.DataFrame, clog: ChangeLog, limit: int = C.INTERP_LIMIT) -> pd.DataFrame:
    """Step 13: fill short gaps of the primary measurements, recompute derived ones.

    Temperature, relative humidity and surface temperature are interpolated;
    absolute humidity and dew point are recomputed with the Magnus formula on
    the filled rows so that all quantities stay physically consistent.
    """
    for s in C.SENSORS:
        any_fill = pd.Series(False, index=df.index)
        for m in ("temp", "rh", "surf_temp"):
            c = f"{m}_{s}"
            if c not in df:
                continue
            df[c], filled = interpolate_short_gaps(df[c], limit)
            any_fill |= filled
            clog.add("interpolate", c, filled.sum(), f"linear, gaps <= {limit} readings", "short transmission gaps")
        t, rh = f"temp_{s}", f"rh_{s}"
        rows = any_fill & df[t].notna() & df[rh].notna()
        df.loc[rows, f"abs_hum_{s}"] = absolute_humidity(df.loc[rows, t], df.loc[rows, rh])
        df.loc[rows, f"dew_point_{s}"] = dew_point(df.loc[rows, t], df.loc[rows, rh])
        df[f"flag_interp_{s}"] = any_fill
    return df


def final_checks(df: pd.DataFrame) -> None:
    """Step 14: assertions that must hold for every later notebook."""
    idx = df.index
    assert idx.is_unique and idx.is_monotonic_increasing, "index not unique / sorted"
    assert str(idx.tz) == C.LOCAL_TZ, "index not in local time zone"
    assert (pd.Series(idx).diff().dropna() == pd.Timedelta(C.FREQ)).all(), "not a regular 20-min grid"
    meas = measurement_columns(df)
    assert not (df[meas] == C.MISSING_CODE).any().any(), "missing code left"
    for s in C.SENSORS:
        assert df[f"rh_{s}"].dropna().between(0, 100).all(), f"rh out of range {s}"
        both = df[f"dew_point_{s}"].notna() & df[f"temp_{s}"].notna()
        assert (df.loc[both, f"dew_point_{s}"] <= df.loc[both, f"temp_{s}"] + 0.01).all(), f"dew point > temp {s}"
    flags = [c for c in df.columns if c.startswith("flag_")]
    assert all(df[c].dtype == bool for c in flags), "flags must be boolean"
    for c in meas:
        lo, hi = C.PLAUSIBLE[measure_of(c)]
        assert df[c].dropna().between(lo, hi).all(), f"{c} implausible"


def clean(raw: pd.DataFrame | None = None, interp_limit: int = C.INTERP_LIMIT):
    """Run the full cleaning pipeline. Returns (clean frame, change log, longest runs)."""
    clog = ChangeLog()
    raw = read_raw() if raw is None else raw
    df = basic_parse(raw, clog)
    df = commissioning(df, clog)
    df = plausibility(df, clog)
    df = physical_fixes(df, clog)
    df = spikes(df, clog)
    df = warmup_artifacts(df, clog)
    df, longest = stuck_values(df, clog)
    df = sun_flag(df, clog)
    df.index = df.index.tz_convert(C.LOCAL_TZ)
    clog.add("timezone", "index", len(df), "UTC -> Europe/Berlin", "local time for daily patterns")
    df = interpolate(df, clog, interp_limit)
    # column order: measurements by sensor, then flags
    meas = [c for s in C.SENSORS for c in sensor_columns(df, s)]
    flags = sorted(c for c in df.columns if c.startswith("flag_"))
    df = df[meas + flags]
    final_checks(df)
    return df, clog.frame(), longest


def load_raw_minimal(path=C.RAW_FILE) -> pd.DataFrame:
    """Raw data with only parsing applied (no fixes, no interpolation).

    Used for robustness checks (for example organ events on raw data).
    """
    df = basic_parse(read_raw(path), ChangeLog())
    df.index = df.index.tz_convert(C.LOCAL_TZ)
    return df


def save(df: pd.DataFrame, clog: pd.DataFrame) -> None:
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.PROCESSED_FILE)
    clog.to_csv(C.CHANGE_LOG_FILE, index=False)


def load_processed(path=C.PROCESSED_FILE) -> pd.DataFrame:
    """Load the cleaned data; runs the pipeline first if the file is missing."""
    if not path.exists():
        main()
    df = pd.read_parquet(path)
    if str(df.index.tz) != C.LOCAL_TZ:
        df.index = df.index.tz_convert(C.LOCAL_TZ)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df, clog, _ = clean()
    save(df, clog)
    log.info("saved %s  shape=%s", C.PROCESSED_FILE, df.shape)


if __name__ == "__main__":
    main()
