# %% [markdown]
# # 01 · Data understanding, ingestion and preprocessing
#
# **Goal:** turn the raw logger export into one clean, documented, analysis-ready table
# (`data/processed/clean.parquet`). All logic lives in `src/load_data.py`; this notebook
# runs it step by step and prints the evidence for every decision.
#
# The complete pipeline can also be run with a single command from the repository root:
# `python -m src.load_data`.

# %%
import sys, pathlib
ROOT = pathlib.Path.cwd().resolve()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config as C, load_data as L, results, plotting as P
from src.physics import absolute_humidity, dew_point

P.setup()
pd.set_option("display.width", 160, "display.max_columns", 50)

# %% [markdown]
# ## 1 · Raw file: schema, size and format
#
# The file is a semicolon-separated InfluxDB export (UTF-8, CRLF line endings) with a
# **decimal comma** and the **missing code `-255`**. We read it with `sep=";"` and
# `decimal=","`, which makes every measurement column numeric without string hacks.

# %%
raw = L.read_raw()
print("shape:", raw.shape)
print("dtypes:", raw.dtypes.value_counts().to_dict())
print("technical columns:", {c: raw[c].unique()[:3].tolist() for c in C.TECHNICAL_COLUMNS})
with open(C.RAW_FILE, "rb") as fh:
    head = fh.read(4000)
print("CRLF line endings:", b"\r\n" in head)
raw.iloc[:3, :8]

# %% [markdown]
# ### Sensor dictionary
#
# Column pattern `<Measure>_<Sensor>`. Measures are renamed to short English prefixes.
# Absolute humidity and dew point are *derived* by the logger from temperature and
# relative humidity (verified below).

# %%
measures = pd.DataFrame([(k, v[0], v[1], v[2]) for k, v in C.MEASURES.items()],
                        columns=["raw name", "short prefix", "unit", "meaning"])
display(measures)
cols = [c for c in raw.columns if c not in C.TECHNICAL_COLUMNS + [C.TIME_COLUMN]]
avail = pd.DataFrame(0, index=list(C.SENSORS), columns=[v[0] for v in C.MEASURES.values()])
for c in cols:
    m, s = c.rsplit("_", 1)
    avail.loc[s, C.MEASURES[m][0]] = 1
sensors = pd.DataFrame(C.SENSORS).T[["client", "level", "has_surface"]].join(avail)
sensors.index.name = "code"
display(sensors)
results.save_table(sensors, "sensor_dictionary")

# %% [markdown]
# ### Time axis
#
# `_time` is ISO-8601 in **UTC**. We parse it as UTC, check duplicates and the
# 20-minute grid, and only at the end convert to **Europe/Berlin** local time, because
# daily patterns (heating, services) follow local clock time. Doing all grid checks in
# UTC avoids the duplicate / missing hour at the daylight-saving switches.

# %%
t = pd.to_datetime(raw[C.TIME_COLUMN], utc=True)
step = t.diff().dropna()
print("first:", t.min(), " last:", t.max(), " rows:", len(t))
print("duplicated timestamps:", t.duplicated().sum())
print("sampling steps:", step.value_counts().to_dict())
print("expected rows on a 20-min grid:", len(pd.date_range(t.min(), t.max(), freq="20min")))
local = t.dt.tz_convert(C.LOCAL_TZ)
print("UTC offsets present in local time:", sorted(local.map(lambda x: x.utcoffset()).astype(str).unique()))

# %% [markdown]
# ## 2 · Missing code and raw value ranges
#
# `-255` is a logger code for "no value". It must become `NaN` *before* any statistic
# is computed; otherwise means and minima are meaningless. We also verify that no other
# sentinel-like value (e.g. `-999`, `0` blocks) exists.

# %%
num = raw[cols]
print("cells equal to -255:", int((num == C.MISSING_CODE).sum().sum()), "of", num.size)
print("values < -100 other than -255:", int(((num < -100) & (num != C.MISSING_CODE)).sum().sum()))
desc = num.replace(C.MISSING_CODE, np.nan).describe().T[["count", "min", "mean", "max"]]
desc.round(2)

# %% [markdown]
# ## 3 · Run the cleaning pipeline
#
# `L.clean()` performs all steps listed in the module docstring and returns the clean
# frame, a **change log** (every modification with count and reason) and the longest run
# of identical values per column. Nothing is changed silently.

# %%
df, clog, longest = L.clean(raw)
print("clean shape:", df.shape, "| index:", df.index.dtype, "| from", df.index.min(), "to", df.index.max())
summary_log = clog[clog.n_values > 0].groupby(["step", "action"])["n_values"].sum().to_frame()
summary_log

# %% [markdown]
# ### 3a · Humidity above 100 % and dew point above air temperature
#
# Only the **outside** sensor reports relative humidity above 100 % (fog, rain, sensor
# saturation). Indoor sensors never do. Air cannot hold more than 100 % relative
# humidity, so we **clip to 100 %**, set the dew point to the air temperature,
# recompute absolute humidity at saturation and keep the flag `flag_rh_clipped_AU`.

# %%
r = raw.replace(C.MISSING_CODE, np.nan)
checks = pd.DataFrame({
    s: {"rh > 100 %": int((r[f"Luft-Feuchtigkeit_{s}"] > 100).sum()),
        "max rh": r[f"Luft-Feuchtigkeit_{s}"].max(),
        "dew point > air temp": int((r[f"Taupunkt_{s}"] > r[f"Luft-Temperatur_{s}"]).sum())}
    for s in C.SENSORS}).T
display(checks)
n_rh, n_dp = int(checks.loc["AU", "rh > 100 %"]), int(checks.loc["AU", "dew point > air temp"])
print(f"outside: {n_rh} readings above 100 % (max {checks.loc['AU','max rh']:.1f} %), {n_dp} with dew point > air temp")
print("after cleaning: max rh outside =", df["rh_AU"].max(), "| dew>temp:", int((df.dew_point_AU > df.temp_AU + 0.01).sum()))
results.update(rh_over_100_AU=n_rh, rh_max_AU=checks.loc["AU", "max rh"], dp_over_t_AU=n_dp)

# %% [markdown]
# ### 3b · Are derived quantities consistent? (Magnus formula)
#
# If absolute humidity and dew point are computed by the logger from temperature and
# relative humidity, we can recompute them and use the same formula later (surface
# relative humidity). Differences within ±0.005 mean the logger uses the Magnus
# constants 17.62 / 243.12 °C and rounds to two decimals.

# %%
rows = []
for s in C.SENSORS:
    T, RH = r[f"Luft-Temperatur_{s}"], r[f"Luft-Feuchtigkeit_{s}"]
    ok = RH <= 100
    rows.append(dict(sensor=s,
                     max_abs_diff_abs_hum=np.nanmax(np.abs(absolute_humidity(T[ok], RH[ok]) - r[f"Absolute-Luft-Feuchtigkeit_{s}"][ok])),
                     max_abs_diff_dew_point=np.nanmax(np.abs(dew_point(T[ok], RH[ok]) - r[f"Taupunkt_{s}"][ok]))))
magnus = pd.DataFrame(rows).set_index("sensor").round(4)
display(magnus)
results.update(magnus_max_diff=float(magnus.max().max()))

# %% [markdown]
# **Decision:** derived values are exactly reproducible, so after interpolating
# temperature and humidity we *recompute* absolute humidity and dew point instead of
# interpolating them separately (keeps the physics consistent).
#
# ### 3c · Sensor warm-up after installation and outages
#
# Right after installation (19 Jan 2024) and after long outages several sensors report a
# value far from the room and relax exponentially within about an hour. Example: the
# organ sensor starts at 24.8 °C and falls to 18.7 °C in one hour, while the church is at
# about 10 °C. This is the sensor adapting, not the room. Left in the data it would create
# false rapid-change events at the organ.

# %%
raw_min = L.load_raw_minimal()
fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
for ax, (s, start) in zip(axes, [("O1", "2024-01-19 09:00"), ("Ke1", "2024-07-26 09:00"), ("AWo", "2025-09-05 09:00")]):
    t0 = pd.Timestamp(start, tz=C.LOCAL_TZ)
    w = raw_min.loc[t0:t0 + pd.Timedelta("8h"), f"temp_{s}"]
    fl = df.loc[w.index, f"flag_warmup_{s}"]
    ax.plot(w.index, w, color=P.SERIES[0], marker="o", ms=3, label="raw reading")
    ax.plot(w.index[fl], w[fl], "o", color=P.STATUS["high"], ms=6, label="removed (warm-up)")
    ax.set_title(f"{P.name(s)}: sensor settles after restart")
    ax.set_ylabel("Air temperature (°C)")
    ax.tick_params(axis="x", rotation=30)
axes[0].legend()
plt.tight_layout(); P.save(fig, "01_warmup.png"); plt.show()
print(clog[clog.step == "warmup"][["column", "n_values", "action"]].to_string(index=False))

# %% [markdown]
# **Decision:** after a sensor (re)start following a gap of at least 6 h, the first
# 6 readings (2 h) are set to `NaN` and flagged (`flag_warmup_*`). This removes
# 195 readings in total (mostly the patchy attic sensor) and is logged.
#
# ### 3d · Stuck sensors
#
# A frozen sensor repeats the same value. We list the longest run of identical values per
# column; runs of ≥ 12 h (36 readings) in temperature or humidity would be set to `NaN`.
# Runs at exactly 100 % outside humidity are saturation (step 3a), not a frozen sensor.

# %%
lr = longest.to_frame()
lr["hours"] = (lr["longest_identical_run"] / 3).round(1)
display(lr.sort_values("longest_identical_run", ascending=False).head(10))
print("columns with a run >= 12 h (excluding derived quantities):",
      [c for c, v in longest.items() if v >= C.STUCK_MIN_RUN and L.measure_of(c) in ("temp", "rh", "surf_temp")])

# %% [markdown]
# **Decision:** the longest runs are ≤ 5.7 h and occur in derived absolute humidity (which
# is stable at 0.01 g/m³ resolution in a calm building) or the very stable basement. No
# sensor is stuck; nothing is removed.
#
# ### 3e · Extreme outside temperatures – direct sun on the sensor?
#
# The outside sensor reaches about 40 °C. The daily maximum of air temperature normally
# occurs around 14–16 h. If the sensor is hit by the low evening sun on the west tower,
# its maximum shifts to the evening and the values exceed realistic air temperatures.

# %%
au = df["temp_AU"]
au_ok = au.dropna()
au_day = au_ok.groupby(au_ok.index.date)
daily_max_hour = au_day.idxmax()[au_day.count() >= 60].map(lambda x: x.hour)  # days with >= 20 h of data
summer = au[(au.index.month >= 5) & (au.index.month <= 8)]
hourly = summer.groupby(summer.index.hour).mean()
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
daily_max_hour.value_counts().sort_index().plot.bar(ax=axes[0], color=P.SERIES[0], width=0.8)
axes[0].set_title("Outside sensor: on half of all days the maximum is in the evening")
axes[0].set_xlabel("Hour of daily maximum (local time)"); axes[0].set_ylabel("Number of days")
axes[1].plot(hourly.index, hourly, marker="o", color=P.SERIES[1])
axes[1].set_title("Summer: mean outside temperature peaks at 19-20 h")
axes[1].set_xlabel("Hour of day (local time)"); axes[1].set_ylabel("Mean temperature (°C)")
plt.tight_layout(); P.save(fig, "01_sun_outside.png"); plt.show()
print("share of days with maximum between 17 and 21 h:", round(daily_max_hour.between(17, 21).mean(), 3))
print("readings > 32 °C:", int((au > 32).sum()), "| of these between 16 and 21 h:", int(df["flag_sun_AU"].sum()))
print("top 5 values:", au.nlargest(5).round(2).to_dict())
results.update(sun_flag_n=int(df["flag_sun_AU"].sum()), au_max=float(au.max()),
               au_max_evening_share=float(daily_max_hour.between(17, 21).mean()))

# %% [markdown]
# **Decision:** on about half of all days the maximum falls between 17 and 21 h, and the summer
# mean profile peaks at 19–20 h. This evening peak (instead of the usual afternoon peak) is the signature of
# direct evening sun on the west-facing sensor. Readings above 32 °C between 16 and 21 h
# are **flagged** (`flag_sun_AU`), not deleted: they are real heat on the facade but do
# not represent the air. They are excluded where the outside *air* matters (lag
# correlation). Indoor sensors are never above 36 °C except the attic, where 45 °C under a
# sun-heated roof is plausible.
#
# ### 3f · Short gaps: interpolation of at most 1 hour
#
# Gaps of up to 3 readings (1 h) are filled linearly in time; longer gaps stay `NaN`.
# Notebook 02 justifies this limit with the gap length distribution, a hold-out test and
# a sensitivity check with limits 2, 3 and 6. Absolute humidity and dew point are
# recomputed on filled rows.

# %%
interp = clog[clog.step == "interpolate"][["column", "n_values"]]
print(interp.to_string(index=False))
print("share of all sensor values that are interpolated:",
      round(interp.n_values.sum() / df[L.measurement_columns(df)].notna().sum().sum(), 5))

# %% [markdown]
# ## 4 · Final assertions and data availability
#
# `final_checks` asserts: unique, sorted, regular 20-minute local-time index; no `-255`
# left; relative humidity within 0–100 %; dew point ≤ air temperature; all values in
# plausible ranges; flags are boolean. The pipeline fails loudly if any of these break.

# %%
L.final_checks(df)
print("all final assertions passed")
temp_cols = [f"temp_{s}" for s in C.SENSORS]
cov = df[temp_cols].notna().groupby(df.index.strftime("%Y-%m")).mean()
cov.columns = [P.name(s) for s in C.SENSORS]
fig, ax = P.calendar_heatmap(cov, "Data availability: wall sensors are missing for most of 2024/25",
                             "Share of readings available", cmap=P.SEQ_BLUE)
P.save(fig, "01_availability.png"); plt.show()
results.save_table(cov.round(3), "availability_by_month")
avail_tot = df[temp_cols].notna().mean().rename(lambda c: c.split("_")[1]).round(3)
print(avail_tot.to_string())
results.update(n_rows_clean=len(df), start=str(df.index.min()), end=str(df.index.max()),
               availability={k: float(v) for k, v in avail_tot.items()},
               n_warmup=int(clog.query("step=='warmup'").n_values.sum()),
               n_interpolated=int(interp.n_values.sum()))

# %% [markdown]
# **Key availability facts (decision for all later notebooks):** organ, south art objects
# and outside air are almost complete. The north wall only covers Jan–Feb 2024 and
# Sep–Nov 2025; the south and east walls have no data from about Oct/Nov 2024 to Sep 2025;
# the north-west art objects stop in early April 2025 (only surface temperature continues
# from Sep 2025, without humidity); the basement misses Mar–Jul 2024. **No wall sensor
# covers the winter 2024/25.** All later comparisons therefore normalise by valid hours
# and use common time windows.
#
# ## 5 · Save

# %%
L.save(df, clog)
print("saved", C.PROCESSED_FILE.relative_to(ROOT), "and", C.CHANGE_LOG_FILE.relative_to(ROOT))
clog[clog.n_values > 0].to_csv(C.TABLE_DIR / "preprocessing_log.csv", index=False)
