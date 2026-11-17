"""Feature engineering for mold risk characterisation.

All features follow the scientific background of the brief:
moisture (relative humidity vs. the LIM curve), temperature (LIM depends on
temperature), surface vs. dew point (condensation) and duration (rolling
windows, episodes, cumulative exposure).

Missing data is never treated as safe: every binary flag is ``NaN`` where its
inputs are missing, and rolling shares are divided by *valid* hours only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .load_data import run_lengths
from .physics import lim_threshold, rh_at_temperature


def as_flag(cond: pd.Series, valid: pd.Series) -> pd.Series:
    """Float flag: 1 where ``cond``, 0 where not, NaN where inputs are invalid."""
    return cond.astype(float).where(valid)


def location_features(df: pd.DataFrame, s: str, lim_class: str = C.DEFAULT_LIM_CLASS,
                      margin_crit: float = C.MARGIN_CRITICAL, basis: str = "auto") -> pd.DataFrame:
    """Per-reading physical features of one location.

    ``basis`` decides which humidity is compared with the LIM curve:

    * ``"auto"`` (default): the surface relative humidity at the measured
      surface temperature where a surface sensor exists (mold grows on
      surfaces, and cold surfaces see higher humidity), otherwise the room air.
    * ``"air"``: room air for every location (used to compare all locations
      on the same footing in the sensitivity analysis).
    """
    out = pd.DataFrame(index=df.index)
    out["air_temp"] = df[f"temp_{s}"]
    out["air_rh"] = df[f"rh_{s}"]
    out["dew_point"] = df[f"dew_point_{s}"]
    out["abs_hum"] = df[f"abs_hum_{s}"]
    has_surface = f"surf_temp_{s}" in df
    if has_surface:
        out["surf_temp"] = df[f"surf_temp_{s}"]
        out["margin"] = out["surf_temp"] - out["dew_point"]          # dew point margin
        out["surf_rh"] = rh_at_temperature(out["dew_point"], out["surf_temp"])
        out["surf_minus_air"] = out["surf_temp"] - out["air_temp"]
    else:
        out["surf_temp"] = out["margin"] = out["surf_rh"] = out["surf_minus_air"] = np.nan

    use_surface = has_surface and basis == "auto"
    out["ref_temp"] = out["surf_temp"] if use_surface else out["air_temp"]
    out["ref_rh"] = out["surf_rh"] if use_surface else out["air_rh"]
    out["lim"] = lim_threshold(out["ref_temp"], lim_class)
    out["lim_excess"] = out["ref_rh"] - out["lim"]                   # > 0 = above the mold limit
    valid = out["ref_rh"].notna() & out["ref_temp"].notna()
    out["crit_lim"] = as_flag(out["lim_excess"] > 0, valid)

    mvalid = out["margin"].notna()
    out["condensation"] = as_flag(out["margin"] <= 0, mvalid)
    out["crit_margin"] = as_flag(out["margin"] <= margin_crit, mvalid)

    # cross-sensor features: internal moisture source vs ventilation potential
    out["d_temp_in_out"] = out["air_temp"] - df["temp_AU"]
    out["d_abs_hum_in_out"] = out["abs_hum"] - df["abs_hum_AU"]
    out["interpolated"] = df.get(f"flag_interp_{s}", pd.Series(False, index=df.index))
    return out


def rolling_exposure(flag: pd.Series, window: str, min_valid_share: float = C.MIN_VALID_SHARE) -> pd.DataFrame:
    """Trailing-window exposure of a 0/1/NaN flag.

    Returns critical hours, valid hours and the critical share
    (critical / valid). The share is NaN when less than ``min_valid_share`` of
    the window has valid data, so that outages cannot look safe.
    """
    per_reading_h = 1.0 / C.READINGS_PER_HOUR
    crit_h = flag.fillna(0).rolling(window).sum() * per_reading_h
    valid_h = flag.notna().astype(float).rolling(window).sum() * per_reading_h
    window_h = pd.Timedelta(window) / pd.Timedelta("1h")
    share = (crit_h / valid_h).where(valid_h >= min_valid_share * window_h)
    return pd.DataFrame({"crit_h": crit_h, "valid_h": valid_h, "share": share})


def episodes(flag: pd.Series) -> pd.DataFrame:
    """Uninterrupted runs of ``flag == 1``. Missing data ends an episode.

    Returns start, end (last critical reading) and duration in hours.
    """
    runs = run_lengths(flag.eq(1))
    if runs.empty:
        return pd.DataFrame(columns=["start", "end", "duration_h"])
    runs["duration_h"] = runs["length"] / C.READINGS_PER_HOUR
    return runs[["start", "end", "duration_h"]].reset_index(drop=True)


def duration_features(feat: pd.DataFrame, flag_col: str = "crit_lim") -> pd.DataFrame:
    """Rolling 24 h / 7 d / 30 d critical hours and shares, cumulative exposure."""
    f = feat[flag_col]
    out = pd.DataFrame(index=feat.index)
    for w in ("24h", "7D", "30D"):
        r = rolling_exposure(f, w)
        out[f"{flag_col}_h_{w}"] = r["crit_h"]
        out[f"{flag_col}_share_{w}"] = r["share"]
    out[f"{flag_col}_cum_h"] = f.fillna(0).cumsum() / C.READINGS_PER_HOUR
    out[f"{flag_col}_cum_valid_h"] = f.notna().cumsum() / C.READINGS_PER_HOUR
    return out


def time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features in local time."""
    out = pd.DataFrame(index=index)
    out["hour"] = index.hour
    out["month"] = index.month
    out["weekday"] = index.weekday               # 0 = Monday, 6 = Sunday
    out["is_sunday"] = index.weekday == 6
    out["season"] = pd.Categorical(
        np.select([index.month.isin([12, 1, 2]), index.month.isin([3, 4, 5]), index.month.isin([6, 7, 8])],
                  ["winter", "spring", "summer"], "autumn"),
        categories=["winter", "spring", "summer", "autumn"])
    out["heating_season"] = index.month.isin(C.HEATING_MONTHS)
    return out


def heating_season_label(index: pd.DatetimeIndex) -> pd.Series:
    """Label like ``'2024/25'`` for Oct-Apr readings, NaN in summer."""
    y = np.where(index.month >= 10, index.year, index.year - 1)
    lab = pd.Series([f"{a}/{str(a + 1)[-2:]}" for a in y], index=index)
    return lab.where(index.month.isin(C.HEATING_MONTHS))


def build_features(df: pd.DataFrame, lim_class: str = C.DEFAULT_LIM_CLASS,
                   margin_crit: float = C.MARGIN_CRITICAL, basis: str = "auto",
                   with_duration: bool = True) -> pd.DataFrame:
    """Long table (time x location) with physical, duration and time features."""
    parts = []
    tf = time_features(df.index)
    for s in C.INDOOR:
        f = location_features(df, s, lim_class, margin_crit, basis)
        if with_duration:
            f = f.join(duration_features(f, "crit_lim"))
        f = f.join(tf)
        f["location"] = s
        parts.append(f)
    out = pd.concat(parts)
    out["location"] = pd.Categorical(out["location"], categories=C.INDOOR)
    return out
