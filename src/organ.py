"""Organ temperature stability: detection of critical one-hour changes.

Critical event (brief): |T(t0) - T(t1)| > 1 degC with t1 - t0 = exactly 1 hour.
The partner value T(t0 + 1 h) is looked up *by time*, so a gap in the data
can never pair two readings that are further apart than one hour.
"""
from __future__ import annotations

import pandas as pd

from . import config as C


def one_hour_change_time(temp: pd.Series, window: str = C.ORGAN_WINDOW) -> pd.Series:
    """T(t0 + window) - T(t0) via exact timestamp lookup (DST-safe, gap-safe)."""
    temp = temp.dropna()
    t1 = temp.index + pd.Timedelta(window)
    later = temp.reindex(t1).to_numpy()
    return pd.Series(later - temp.to_numpy(), index=temp.index, name="delta").dropna()


def one_hour_change_rowshift(temp: pd.Series, rows: int = C.READINGS_PER_HOUR, dropna: bool = True) -> pd.Series:
    """Naive version: shift by a fixed number of rows.

    With ``dropna=True`` (what a naive analysis of a gappy series does) the
    shift silently pairs readings across gaps; it is kept only to show why the
    time-based version is needed.
    """
    s = temp.dropna() if dropna else temp
    return (s.shift(-rows) - s).rename("delta").dropna()


def detect_events(temp: pd.Series, threshold: float = C.ORGAN_DELTA_T, method: str = "time") -> pd.DataFrame:
    """All start readings t0 with |T(t0+1h) - T(t0)| > threshold."""
    delta = one_hour_change_time(temp) if method == "time" else one_hour_change_rowshift(temp)
    ev = delta[delta.abs() > threshold].to_frame()
    ev["t1"] = ev.index + pd.Timedelta(C.ORGAN_WINDOW)
    ev["direction"] = ev["delta"].gt(0).map({True: "warming", False: "cooling"})
    ev.index.name = "t0"
    return ev


def merge_episodes(events: pd.DataFrame, max_gap: str = C.ORGAN_WINDOW) -> pd.DataFrame:
    """Merge events whose start times are at most ``max_gap`` apart into episodes.

    Default 1 h: overlapping one-hour windows describe the same physical
    temperature swing. A stricter variant (``max_gap='20min'``) only merges
    directly consecutive readings.
    """
    if events.empty:
        return pd.DataFrame(columns=["start", "end", "n_events", "max_abs_delta", "direction"])
    t = events.index.to_series()
    new = t.diff() > pd.Timedelta(max_gap)
    ep_id = new.cumsum()
    g = events.assign(ep=ep_id.values, absd=events["delta"].abs()).groupby("ep")
    out = pd.DataFrame({
        "start": g.apply(lambda x: x.index.min()),
        "end": g["t1"].max(),
        "n_events": g.size(),
        "max_abs_delta": g["absd"].max(),
        "net_delta": g["delta"].sum() / g.size(),
        "direction": g["direction"].agg(lambda x: x.mode().iloc[0] if x.nunique() == 1 else "mixed"),
    })
    out["duration_h"] = (out["end"] - out["start"]) / pd.Timedelta("1h")
    return out.reset_index(drop=True)
