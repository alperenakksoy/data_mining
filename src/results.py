"""Tiny key-value store for numbers that the report cites (report/tables/results.json)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C


def _plain(v):
    """Convert numpy / pandas scalars to JSON-serialisable Python values."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def update(**kwargs) -> None:
    """Merge values into results.json."""
    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    data = load()
    data.update({k: _plain(v) for k, v in kwargs.items()})
    C.RESULTS_JSON.write_text(json.dumps(data, indent=1, sort_keys=True))


def load() -> dict:
    if C.RESULTS_JSON.exists():
        return json.loads(C.RESULTS_JSON.read_text())
    return {}


def save_table(df: pd.DataFrame, name: str) -> None:
    """Save a table as CSV in report/tables for the report."""
    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TABLE_DIR / f"{name}.csv")
