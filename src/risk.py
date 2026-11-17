"""Transparent rule-based mold risk indicator and its validation helpers.

Risk levels per reading and location
-----------------------------------
* **high (2)**: condensation (dew point margin <= 0 degC) **or** humidity above
  the LIM curve now *and* during at least ``PERSIST_SHARE`` of the valid time
  of the trailing ``PERSIST_WINDOW`` (moisture that persists).
* **medium (1)**: humidity above the LIM curve **or** dew point margin in
  (0, ``MARGIN_CRITICAL``] degC.
* **low (0)**: otherwise.
* **NaN**: required inputs missing. Missing is never counted as low.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config as C
from .features import episodes, location_features, rolling_exposure

LEVELS = {0: "low", 1: "medium", 2: "high"}
RISK_COLORS = {0: "#4c9a2a", 1: "#f2c12e", 2: "#c8302b"}  # green / yellow / red


def classify(feat: pd.DataFrame, margin_crit: float = C.MARGIN_CRITICAL,
             persist_window: str = C.PERSIST_WINDOW, persist_share: float = C.PERSIST_SHARE) -> pd.Series:
    """Risk level (0/1/2, NaN = unknown) for one location's feature frame."""
    valid = feat["crit_lim"].notna()
    above = feat["crit_lim"].eq(1)
    persist = rolling_exposure(feat["crit_lim"], persist_window)["share"] >= persist_share
    margin = feat["margin"]
    cond = margin <= 0
    near = (margin > 0) & (margin <= margin_crit)
    high = cond | (above & persist)
    medium = above | near
    level = np.select([high, medium], [2.0, 1.0], 0.0)
    return pd.Series(level, index=feat.index, name="risk").where(valid | margin.notna())


def risk_table(df: pd.DataFrame, lim_class: str = C.DEFAULT_LIM_CLASS, margin_crit: float = C.MARGIN_CRITICAL,
               persist_window: str = C.PERSIST_WINDOW, persist_share: float = C.PERSIST_SHARE,
               basis: str = "auto") -> pd.DataFrame:
    """Wide table time x location of risk levels."""
    out = {}
    for s in C.INDOOR:
        f = location_features(df, s, lim_class, margin_crit, basis)
        out[s] = classify(f, margin_crit, persist_window, persist_share)
    return pd.DataFrame(out)


def summarize(risk: pd.DataFrame, mask: pd.Series | None = None) -> pd.DataFrame:
    """Per-location shares (of *valid* hours), episodes and a ranking score.

    ``mask`` restricts the evaluation to a common time window (boolean index).
    ``score`` = share high + 0.5 * share medium (time-weighted exposure).
    """
    rows = []
    for s in risk.columns:
        r = risk[s] if mask is None else risk[s].where(mask)
        valid = r.notna()
        n = valid.sum()
        hi_ep = episodes((r == 2).astype(float).where(valid))
        el_ep = episodes((r >= 1).astype(float).where(valid))
        rows.append(dict(
            location=s,
            valid_h=n / C.READINGS_PER_HOUR,
            coverage=n / len(r) if mask is None else n / max(mask.sum(), 1),
            share_high=(r == 2).sum() / n if n else np.nan,
            share_medium=(r == 1).sum() / n if n else np.nan,
            share_low=(r == 0).sum() / n if n else np.nan,
            n_high_episodes=len(hi_ep),
            longest_high_h=hi_ep["duration_h"].max() if len(hi_ep) else 0.0,
            n_elevated_episodes=len(el_ep),
            longest_elevated_h=el_ep["duration_h"].max() if len(el_ep) else 0.0,
        ))
    out = pd.DataFrame(rows).set_index("location")
    out["score"] = out["share_high"] + 0.5 * out["share_medium"]
    out["rank"] = out["score"].rank(ascending=False, method="min")
    return out.sort_values("score", ascending=False)


def monthly_profile(risk: pd.DataFrame, level: int = 2, at_least: bool = False) -> pd.DataFrame:
    """Share of valid readings per calendar month (YYYY-MM) at a risk level."""
    hit = (risk >= level) if at_least else (risk == level)
    hit = hit.astype(float).where(risk.notna())
    key = risk.index.strftime("%Y-%m")
    return hit.groupby(key).mean()


def common_mask(risk: pd.DataFrame, locations) -> pd.Series:
    """Readings where *all* given locations have a valid risk level."""
    return risk[list(locations)].notna().all(axis=1)


def rank_correlation(a: pd.Series, b: pd.Series) -> float:
    """Spearman correlation of two rankings (index = location)."""
    common = a.index.intersection(b.index)
    return float(spearmanr(a[common], b[common]).statistic)
