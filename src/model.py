"""Optional early-warning model: "will high mold risk occur in the next 24 hours?".

Design choices (to avoid circular validation):

* The target is *future* risk (the next 24 h), not the current rule output.
* Inputs are current outside conditions, the current indoor state and time
  features, all available at prediction time.
* The data is split **by time** (train before the split date, test after),
  never randomly, because neighbouring readings are almost identical.
* A shallow decision tree (depth 3) keeps the model readable.
* It is compared with a naive *persistence* baseline ("high risk in the next
  24 h if the location is at high risk now") and a simple rule of thumb
  ("warn whenever humidity is above the mold limit now").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier, export_text

from . import config as C
from .features import location_features, rolling_exposure

HORIZON = "24h"
FEATURES = [
    "out_temp", "out_abs_hum", "out_temp_mean_24h", "out_temp_change_24h", "out_abs_hum_mean_24h",
    "in_temp", "in_rh", "lim_excess", "crit_share_24h", "crit_share_7d", "d_abs_hum_in_out",
    "high_now", "month", "heating_season",
]


def future_any(flag: pd.Series, horizon: str = HORIZON) -> pd.Series:
    """1 if ``flag == 1`` at any time in (t, t + horizon], NaN if the window has no valid data."""
    # the grid is regular (checked in preprocessing), so a forward window of n rows = horizon
    n = int(pd.Timedelta(horizon) / pd.Timedelta(C.FREQ))
    hit = flag.eq(1).astype(float)
    valid = flag.notna().astype(float)
    fut_hit = hit[::-1].rolling(n, min_periods=1).max()[::-1].shift(-1)
    fut_valid = valid[::-1].rolling(n, min_periods=1).sum()[::-1].shift(-1)
    return fut_hit.where(fut_valid >= n * C.MIN_VALID_SHARE)


def build_dataset(df: pd.DataFrame, risk: pd.Series, location: str) -> pd.DataFrame:
    """Feature matrix + target for one location, sampled hourly."""
    f = location_features(df, location)
    X = pd.DataFrame(index=df.index)
    X["out_temp"] = df["temp_AU"]
    X["out_abs_hum"] = df["abs_hum_AU"]
    X["out_temp_mean_24h"] = df["temp_AU"].rolling("24h", min_periods=36).mean()
    X["out_temp_change_24h"] = df["temp_AU"] - df["temp_AU"].shift(freq="24h").reindex(df.index)
    X["out_abs_hum_mean_24h"] = df["abs_hum_AU"].rolling("24h", min_periods=36).mean()
    X["in_temp"] = f["air_temp"]
    X["in_rh"] = f["air_rh"]
    X["lim_excess"] = f["lim_excess"]
    X["crit_share_24h"] = rolling_exposure(f["crit_lim"], "24h")["share"]
    X["crit_share_7d"] = rolling_exposure(f["crit_lim"], "7D")["share"]
    X["d_abs_hum_in_out"] = f["d_abs_hum_in_out"]
    X["high_now"] = risk.eq(2).astype(float).where(risk.notna())
    X["month"] = df.index.month
    X["heating_season"] = df.index.month.isin(C.HEATING_MONTHS).astype(int)
    X["target"] = future_any(risk.eq(2).astype(float).where(risk.notna()))
    X = X[df.index.minute == 0]  # hourly sampling
    return X.dropna()


def evaluate(y_true, y_pred) -> dict:
    """Precision, recall, F1 and confusion matrix for the positive (high risk) class."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return dict(
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        tn=int(cm[0, 0]), fp=int(cm[0, 1]), fn=int(cm[1, 0]), tp=int(cm[1, 1]),
        n=int(len(y_true)), positives=int(np.sum(y_true)),
    )


OUTSIDE_ONLY = ["out_temp", "out_abs_hum", "out_temp_mean_24h", "out_temp_change_24h",
                "out_abs_hum_mean_24h", "in_temp", "in_rh", "d_abs_hum_in_out", "month", "heating_season"]


def fit_and_evaluate(data: pd.DataFrame, split: str, max_depth: int = 3,
                     features: list[str] | None = None) -> dict:
    """Train a shallow tree before ``split`` and evaluate after it.

    Returns metrics for the model and the persistence baseline on the full
    test set and on the *onset* subset (hours that are not high-risk now),
    where early warning is actually useful.
    """
    features = features or FEATURES
    ts = pd.Timestamp(split, tz=C.LOCAL_TZ)
    train, test = data[data.index < ts], data[data.index >= ts - pd.Timedelta(0)]
    # purge: drop the last 24 h of train, their target looks into the test period
    train = train[train.index < ts - pd.Timedelta(HORIZON)]
    clf = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced",
                                 min_samples_leaf=48, random_state=C.RANDOM_SEED)
    clf.fit(train[features], train["target"].astype(int))
    pred = pd.Series(clf.predict(test[features]), index=test.index)
    base = test["high_now"].astype(int)
    base2 = (test["lim_excess"] > 0).astype(int)
    y = test["target"].astype(int)
    onset = test["high_now"] == 0
    return dict(
        model=clf,
        tree_text=export_text(clf, feature_names=features, decimals=2),
        importances=pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False),
        train_period=(train.index.min(), train.index.max(), len(train), float(train["target"].mean())),
        test_period=(test.index.min(), test.index.max(), len(test), float(test["target"].mean())),
        model_all=evaluate(y, pred), baseline_all=evaluate(y, base),
        above_all=evaluate(y, base2),
        model_onset=evaluate(y[onset], pred[onset]), baseline_onset=evaluate(y[onset], base[onset]),
        above_onset=evaluate(y[onset], base2[onset]),
        predictions=pd.DataFrame({"y": y, "model": pred, "baseline": base, "above_now": base2}),
    )
