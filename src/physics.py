"""Psychrometric helper functions (Magnus formula) and the LIM mold isopleths.

The constants 6.112 hPa / 17.62 / 243.12 degC are the Magnus parameters over
water (Sonntag 1990). The data logger evidently uses the same constants: the
absolute humidity and dew point columns of the raw file are reproduced to
within +/-0.005 (rounding), which we verify in notebook 01.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_LIM_CLASS, LIM_POINTS

A, B, E0 = 17.62, 243.12, 6.112  # Magnus constants, E0 in hPa


def saturation_vapor_pressure(temp):
    """Saturation vapour pressure in hPa at air temperature ``temp`` (degC)."""
    temp = np.asarray(temp, dtype=float)
    return E0 * np.exp(A * temp / (B + temp))


def absolute_humidity(temp, rh):
    """Absolute humidity in g/m3 from temperature (degC) and relative humidity (%)."""
    temp = np.asarray(temp, dtype=float)
    e = np.asarray(rh, dtype=float) / 100.0 * saturation_vapor_pressure(temp)
    return 216.7 * e / (273.15 + temp)


def dew_point(temp, rh):
    """Dew point in degC from temperature (degC) and relative humidity (%)."""
    rh = np.clip(np.asarray(rh, dtype=float), 1e-6, None)
    temp = np.asarray(temp, dtype=float)
    g = np.log(rh / 100.0) + A * temp / (B + temp)
    return B * g / (A - g)


def rh_at_temperature(dew_point_c, temp):
    """Relative humidity (%) that air with dew point ``dew_point_c`` has at ``temp``.

    Used for the *surface relative humidity*: the same moist air touching a
    colder surface has a higher relative humidity than the room air.
    Values are capped at 100 % (condensation).
    """
    rh = 100.0 * saturation_vapor_pressure(dew_point_c) / saturation_vapor_pressure(temp)
    return np.minimum(rh, 100.0)


def lim_threshold(temp, lim_class: str = DEFAULT_LIM_CLASS):
    """Temperature-dependent LIM relative humidity limit (%) by linear interpolation.

    Values outside the read range (0-30 degC) are held constant at the edge
    value. Below 0 degC the curve continues to rise in reality, so holding the
    0 degC value is conservative (flags more risk, not less).
    """
    pts = LIM_POINTS[lim_class]
    xs = np.array(sorted(pts))
    ys = np.array([pts[x] for x in xs])
    out = np.interp(np.asarray(temp, dtype=float), xs, ys)
    if isinstance(temp, pd.Series):
        return pd.Series(out, index=temp.index, name=f"lim_{lim_class}")
    return out
