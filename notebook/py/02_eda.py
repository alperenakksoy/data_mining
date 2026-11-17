# %% [markdown]
# # 02 · Data quality and exploratory analysis
#
# Every section ends with a **Decision** that states what we do with the finding and why.

# %%
import sys, pathlib
ROOT = pathlib.Path.cwd().resolve()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src import config as C, load_data as L, results, plotting as P, organ as O, risk as R

P.setup()
pd.set_option("display.width", 160, "display.max_columns", 40)
rng = np.random.default_rng(C.RANDOM_SEED)
df = L.load_processed()
raw = L.load_raw_minimal()          # parsed only, -255 -> NaN, no other change
print(df.shape, df.index.min(), "->", df.index.max())

# %% [markdown]
# ## 1 · Missing data
#
# ### 1a · How much is missing, per sensor and over time?

# %%
temp_cols = [f"temp_{s}" for s in C.SENSORS]
miss = pd.DataFrame({
    "location": [P.name(s) for s in C.SENSORS],
    "missing share (raw)": raw[temp_cols].isna().mean().values,
    "missing share (clean)": df[temp_cols].isna().mean().values,
}, index=list(C.SENSORS)).round(3)
display(miss)

daily_cov = df[temp_cols].notna().resample("W").mean()
daily_cov.index = daily_cov.index.strftime("%Y-%m-%d")
daily_cov.columns = list(C.SENSORS)
fig, ax = P.calendar_heatmap(daily_cov, "Missing data is structured: long sensor outages, not random gaps",
                             "Share of readings available (weekly)", cmap=P.SEQ_BLUE)
P.save(fig, "02_missing_heatmap.png"); plt.show()

# %% [markdown]
# ### 1b · Length distribution of gaps
#
# The length of the gaps decides whether interpolation is acceptable. We count every
# interior gap in the *raw* temperature series of each sensor.

# %%
gl = {s: L.gap_lengths(raw[f"temp_{s}"]) for s in C.SENSORS}
bins = [0, 1, 2, 3, 6, 18, 72, 504, 10**6]
labels = ["1", "2", "3", "4-6", "7-18 (≤6 h)", "19-72 (≤1 d)", "73-504 (≤1 wk)", "> 1 week"]
gap_tab = pd.DataFrame({P.name(s): pd.cut(g, bins, labels=labels).value_counts().reindex(labels)
                        for s, g in gl.items()}).fillna(0).astype(int)
gap_tab.index.name = "gap length (readings of 20 min)"
display(gap_tab)
allg = pd.concat(gl.values())
share_le3 = (allg <= 3).mean()
missing_in_short = allg[allg <= 3].sum() / allg.sum()
print(f"gaps in total: {len(allg)} | share of gaps <= 3 readings: {share_le3:.1%} "
      f"| but they hold only {missing_in_short:.2%} of all missing readings")
results.save_table(gap_tab, "gap_lengths")
results.update(n_gaps=len(allg), share_gaps_le3=share_le3, missing_in_short_gaps=missing_in_short)

fig, ax = plt.subplots(figsize=(9, 3.4))
vals = np.sort(allg.values)
ax.hist(np.log10(vals), bins=40, color=P.SERIES[0])
ax.axvline(np.log10(3.5), color=P.STATUS["high"], ls="--")
ax.text(np.log10(3.5) + 0.05, ax.get_ylim()[1] * 0.85, "interpolation limit\n(1 hour)", color=P.STATUS["high"])
ticks = [1, 3, 18, 72, 504, 4320]
ax.set_xticks(np.log10(ticks), ["20 min", "1 h", "6 h", "1 day", "1 week", "2 months"])
ax.set_xlabel("Gap length (log scale)"); ax.set_ylabel("Number of gaps")
ax.set_title("Most gaps are a single missing reading; a few outages last months")
P.save(fig, "02_gap_lengths.png"); plt.show()

# %% [markdown]
# ### 1c · Is 1 hour a good interpolation limit? Hold-out test
#
# We take complete stretches of data, delete artificial gaps of known length, interpolate
# them linearly and compare with the true values. If the error at 3 readings is still
# small compared with the effects we analyse (≥ 1 °C organ events, LIM margins of several
# % relative humidity), interpolation up to 1 h is harmless.

# %%
def holdout_error(s: pd.Series, length: int, n: int = 400) -> float:
    """Mean absolute error of linear interpolation over artificial gaps of ``length`` readings."""
    x = s.to_numpy()
    ok = np.where(pd.Series(x).rolling(length + 2).count().to_numpy() == length + 2)[0]  # fully valid windows
    starts = rng.choice(ok, size=min(n, len(ok)), replace=False) - length - 1
    errs = []
    for a in starts:
        left, right = x[a], x[a + length + 1]
        true = x[a + 1: a + length + 1]
        est = left + (right - left) * np.arange(1, length + 1) / (length + 1)
        errs.append(np.abs(est - true).mean())
    return float(np.mean(errs))

lengths = [1, 2, 3, 6, 9, 18]
targets = {"Organ air temp (°C)": raw["temp_O1"], "Organ humidity (%)": raw["rh_O1"],
           "Basement humidity (%)": raw["rh_Ke1"], "Outside temp (°C)": raw["temp_AU"],
           "Outside humidity (%)": raw["rh_AU"].clip(upper=100)}
hold = pd.DataFrame({k: [holdout_error(v, L_) for L_ in lengths] for k, v in targets.items()},
                    index=pd.Index(lengths, name="gap length (readings)")).round(3)
display(hold)
results.save_table(hold, "interp_holdout")
results.update(holdout_organ_temp_3=hold.loc[3, "Organ air temp (°C)"], holdout_organ_rh_3=hold.loc[3, "Organ humidity (%)"],
               holdout_out_temp_3=hold.loc[3, "Outside temp (°C)"], holdout_out_temp_18=hold.loc[18, "Outside temp (°C)"])

# %% [markdown]
# ### 1d · Sensitivity of results to the interpolation limit (2, 3, 6 readings)
#
# We rerun the complete pipeline with limits of 2, 3 and 6 readings and recompute the two
# headline results: organ events and the mold-risk shares.

# %%
raw_file = L.read_raw()
sens = []
for lim in [0, 2, 3, 6]:
    d, clog_k, _ = L.clean(raw_file, interp_limit=lim)
    ev = O.detect_events(d["temp_O1"])
    summ = R.summarize(R.risk_table(d))
    sens.append(dict(limit=lim, filled=int(clog_k.query("step=='interpolate'").n_values.sum()),
                     organ_events=len(ev), organ_episodes=len(O.merge_episodes(ev)),
                     **{f"high share {s}": round(summ.loc[s, "share_high"], 4) for s in ["Ke1", "K1", "AWn", "O1"]},
                     ranking=" > ".join(summ.index)))
sens = pd.DataFrame(sens).set_index("limit")
display(sens)
results.save_table(sens, "interp_sensitivity")

# %% [markdown]
# **Decision (interpolation):** 97 % of all gaps are at most one hour long but they hold
# only a tiny share of the missing readings; the missing data is dominated by a handful of
# outages lasting weeks to months. Over 3 readings the interpolation error is a few
# hundredths of a degree and below 0.4 % relative humidity, far below the 1 °C organ
# threshold and the LIM margins, while it grows clearly for 6-hour gaps (outside
# temperature). Organ events, risk shares and the ranking are practically identical for
# limits 0, 2, 3 and 6. We therefore keep **1 hour** (3 readings): it repairs transmission
# hiccups without inventing data across real outages. Long gaps remain `NaN` and are never
# counted as safe.
#
# ## 2 · Sanity checks (printed evidence)

# %%
rows = []
for s in C.SENSORS:
    for m in ["temp", "rh", "surf_temp", "dew_point", "abs_hum"]:
        c = f"{m}_{s}"
        if c in df:
            x = df[c].dropna()
            jump = df[c].diff().abs()
            rows.append(dict(column=c, n=len(x), min=x.min(), p01=x.quantile(.001), median=x.median(),
                             p999=x.quantile(.999), max=x.max(), max_20min_change=jump.max(),
                             n_change_gt_3=int((jump > 3).sum())))
sanity = pd.DataFrame(rows).set_index("column").round(2)
display(sanity)
results.save_table(sanity, "sanity_checks")
print("relative humidity within 0-100 % everywhere:",
      all(df[f"rh_{s}"].dropna().between(0, 100).all() for s in C.SENSORS))
print("dew point <= air temperature everywhere:",
      all((df[f"dew_point_{s}"] <= df[f"temp_{s}"] + 0.01)[df[f"temp_{s}"].notna() & df[f"dew_point_{s}"].notna()].all() for s in C.SENSORS))
print("surface temperature below air temperature on average:",
      {s: round((df[f"surf_temp_{s}"] - df[f"temp_{s}"]).mean(), 2) for s in C.SURFACE_SENSORS})

# %% [markdown]
# ### Implausible jumps
#
# After removing warm-up readings, the remaining large 20-minute jumps are concentrated in
# the attic (morning sun on the roof) and the upper south art-object sensor (midday sun
# through the south window). They are physically real temperature changes, so they are
# kept. The few large jumps elsewhere are inspected below.

# %%
for c in ["temp_O1", "temp_K2", "temp_D1", "surf_temp_K1"]:
    j = df[c].diff().abs()
    top = j.nlargest(3)
    print(c, "largest 20-min changes:", {str(k): round(v, 2) for k, v in top.items()})
k2_hot = df["temp_K2"] > 28
print("south art objects > 28 °C: hours of day", sorted(set(df.index[k2_hot].hour)),
      "months", sorted(set(df.index[k2_hot].month)))

# %% [markdown]
# **Decision (sanity):** all values are within physical limits after the documented
# fixes. The 9.2 °C single-step change of the outside sensor and the large organ/basement
# jumps were warm-up artefacts and are already removed. Remaining sharp jumps in the attic
# and at the upper south sensor happen only at sunny hours (attic 8 h, south 12–15 h in
# summer) and are kept as real solar heat load. The outside sensor's evening sun readings are
# flagged (notebook 01).
#
# ## 3 · Time series, distributions and patterns

# %%
daily = df[temp_cols].resample("D").mean()
fig, ax = plt.subplots(figsize=(13, 4.2))
for s in C.SENSORS:
    lw = 2.2 if s in ("AU", "O1") else 1.2
    ax.plot(daily.index, daily[f"temp_{s}"], label=P.name(s), color=P.LOCATION_COLORS[s], lw=lw)
ax.set_ylabel("Daily mean air temperature (°C)")
ax.set_title("Indoor temperatures follow the seasons, damped and delayed compared with outside")
ax.legend(ncol=5, fontsize=8.5, loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
P.save(fig, "02_daily_temp.png"); plt.show()

rh_cols = [f"rh_{s}" for s in C.SENSORS]
daily_rh = df[rh_cols].resample("D").mean()
fig, ax = plt.subplots(figsize=(13, 4.2))
for s in C.SENSORS:
    ax.plot(daily_rh.index, daily_rh[f"rh_{s}"], label=P.name(s), color=P.LOCATION_COLORS[s],
            lw=2.2 if s in ("Ke1", "K1") else 1.1)
ax.set_ylabel("Daily mean relative humidity (%)")
ax.set_title("Basement and north-west art objects are the most humid indoor places")
ax.legend(ncol=5, fontsize=8.5, loc="lower left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
P.save(fig, "02_daily_rh.png"); plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
order = C.INDOOR + ["AU"]
for ax, m, lab in [(axes[0], "temp", "Air temperature (°C)"), (axes[1], "rh", "Relative humidity (%)")]:
    data = [df[f"{m}_{s}"].dropna() for s in order]
    bp = ax.boxplot(data, orientation="horizontal", whis=(1, 99), showfliers=False, patch_artist=True)
    for patch, s in zip(bp["boxes"], order):
        patch.set_facecolor(P.LOCATION_COLORS[s]); patch.set_alpha(0.7)
    ax.set_yticks(range(1, len(order) + 1), [P.name(s) for s in order])
    ax.set_xlabel(lab)
axes[0].set_title("Distributions: the attic is the most extreme indoor space")
axes[1].set_title("Basement humidity is high almost all the time")
plt.tight_layout(); P.save(fig, "02_distributions.png"); plt.show()
dist = pd.DataFrame({f"{m}_{q}": df[[f"{m}_{s}" for s in order]].quantile(q).values
                     for m in ["temp", "rh"] for q in [0.05, 0.5, 0.95]}, index=order).round(1)
display(dist)

# %% [markdown]
# ### Seasonal (month) and daily (hour) pattern

# %%
monthly = df[temp_cols + rh_cols].groupby(df.index.month).mean()
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for s in C.SENSORS:
    axes[0].plot(monthly.index, monthly[f"temp_{s}"], marker="o", ms=4, label=P.name(s), color=P.LOCATION_COLORS[s])
    axes[1].plot(monthly.index, monthly[f"rh_{s}"], marker="o", ms=4, label=P.name(s), color=P.LOCATION_COLORS[s])
axes[0].set_ylabel("Mean air temperature (°C)"); axes[1].set_ylabel("Mean relative humidity (%)")
for ax in axes:
    ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]); ax.set_xlabel("Month")
axes[0].set_title("Seasonal cycle: indoors stays between about 7 and 22 °C")
axes[1].set_title("Indoor humidity peaks in late summer and autumn")
axes[1].legend(fontsize=8, ncol=2)
plt.tight_layout(); P.save(fig, "02_monthly.png"); plt.show()

# %%
heat = df[df.index.month.isin(C.HEATING_MONTHS)]
summer = df[df.index.month.isin([6, 7, 8])]
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=False)
for ax, (lab, d) in zip(axes[:2], [("heating season (Oct-Apr)", heat), ("summer (Jun-Aug)", summer)]):
    for s in ["AU", "O1", "K2", "Ke1", "D1"]:
        h = d[f"temp_{s}"].groupby(d.index.hour).mean()
        ax.plot(h.index, h - h.mean(), label=P.name(s), color=P.LOCATION_COLORS[s])
    ax.set_title(f"Daily cycle, {lab}"); ax.set_xlabel("Hour of day"); ax.set_ylabel("Deviation from daily mean (°C)")
axes[0].legend(fontsize=8)
# weekday vs Sunday at the organ, heating season
org = heat["temp_O1"]
dev = org - org.groupby(org.index.date).transform("mean")
for lab, mask, col in [("Monday-Saturday", heat.index.weekday < 6, P.SERIES[0]), ("Sunday", heat.index.weekday == 6, P.SERIES[1])]:
    h = dev[mask].groupby(dev[mask].index.hour).mean()
    axes[2].plot(h.index, h, label=lab, color=col, marker="o", ms=3)
axes[2].set_title("Organ (heating season): Sunday vs other days"); axes[2].set_xlabel("Hour of day")
axes[2].set_ylabel("Deviation from daily mean (°C)"); axes[2].legend()
plt.tight_layout(); P.save(fig, "02_daily_cycle.png"); plt.show()

amp = pd.DataFrame({P.name(s): [d[f"temp_{s}"].groupby(d.index.hour).mean().pipe(lambda h: h.max() - h.min())
                                for d in (heat, summer)] for s in C.SENSORS},
                   index=["heating season", "summer"]).round(2)
print("Amplitude of the mean daily cycle (°C):"); display(amp)
wk = heat.groupby(heat.index.weekday)["temp_O1"].agg(["mean", "std"]).round(2)
wk.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
print("Organ temperature by weekday (heating season):"); display(wk.T)

# %% [markdown]
# ### Indoor vs outdoor

# %%
io = pd.DataFrame({
    "mean temp inside - outside (°C)": [(df[f"temp_{s}"] - df["temp_AU"]).mean() for s in C.INDOOR],
    "winter (DJF)": [(df[f"temp_{s}"] - df["temp_AU"])[df.index.month.isin([12, 1, 2])].mean() for s in C.INDOOR],
    "summer (JJA)": [(df[f"temp_{s}"] - df["temp_AU"])[df.index.month.isin([6, 7, 8])].mean() for s in C.INDOOR],
    "mean abs. humidity inside - outside (g/m³)": [(df[f"abs_hum_{s}"] - df["abs_hum_AU"]).mean() for s in C.INDOOR],
    "share of time outside air is drier (abs.)": [(df[f"abs_hum_{s}"] > df["abs_hum_AU"])[df[f"abs_hum_{s}"].notna()].mean() for s in C.INDOOR],
}, index=[P.name(s) for s in C.INDOOR]).round(2)
display(io)
results.save_table(io, "indoor_outdoor")

# %% [markdown]
# **Decisions (patterns):**
# * The church is a slow, heavy building: indoor daily cycles are small (organ well below
#   1 °C on average) while outside swings by several degrees. **Short-term outside weather
#   matters little; season matters a lot** → mold risk features use rolling windows of
#   days to weeks and the month / heating-season flags.
# * Indoor air is warmer than outside in winter and cooler in summer; absolute humidity
#   indoors is close to outside on average → **no strong internal moisture source**; the
#   humidity mainly comes in from outside. In summer, warm humid outside air entering the
#   cool church (and the even cooler basement) raises the relative humidity → summer
#   ventilation can *increase* risk. This motivates the ventilation rule "only ventilate
#   when the outside absolute humidity is lower than inside".
# * On Sundays the organ shows a morning warm-up (services), but rapid changes are not
#   limited to Sundays (see notebook 05).
#
# ## 4 · Cross-sensor correlation and lagged response to outside temperature

# %%
dm = df[temp_cols + rh_cols].resample("D").mean()
corr = dm[temp_cols].corr()
corr.index = corr.columns = [P.name(s) for s in C.SENSORS]
corr_rh = dm[rh_cols].corr()
corr_rh.index = corr_rh.columns = [P.name(s) for s in C.SENSORS]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, cm, t in [(axes[0], corr, "Daily temperatures are almost perfectly correlated"),
                  (axes[1], corr_rh, "Daily humidity is less uniform across the building")]:
    im = ax.imshow(cm, cmap=P.DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cm)), cm.columns, rotation=60, ha="right"); ax.set_yticks(range(len(cm)), cm.index)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if not np.isnan(cm.iloc[i, j]):
                ax.text(j, i, f"{cm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7.5)
    ax.grid(False); ax.set_title(t)
fig.colorbar(im, ax=axes, fraction=0.02, label="Pearson correlation (daily means)")
P.save(fig, "02_correlation.png"); plt.show()

# %% [markdown]
# The seasonal cycle dominates any correlation of raw temperatures (everything is warm in
# summer). To measure how fast the building *reacts* to weather, we remove the seasonal
# component (deviation from a centred 7-day mean) and correlate the outside temperature at
# time *t* with each indoor temperature at *t + lag* for lags 0–48 h (hourly means; outside
# readings flagged as direct sun are excluded).

# %%
hourly = df[temp_cols].where(~df["flag_sun_AU"].to_numpy()[:, None] | (np.array(temp_cols) != "temp_AU")).resample("h").mean()
anom = hourly - hourly.rolling(24 * 7, center=True, min_periods=24 * 4).mean()
lags = range(0, 49)
lag_corr = pd.DataFrame({s: [anom["temp_AU"].corr(anom[f"temp_{s}"].shift(-k)) for k in lags] for s in C.INDOOR}, index=lags)
fig, ax = plt.subplots(figsize=(10, 4))
for s in C.INDOOR:
    ax.plot(lag_corr.index, lag_corr[s], label=P.name(s), color=P.LOCATION_COLORS[s])
    k = lag_corr[s].idxmax(); ax.plot(k, lag_corr[s].max(), "o", color=P.LOCATION_COLORS[s])
ax.set_xlabel("Hours after the outside temperature change"); ax.set_ylabel("Correlation (weather anomaly)")
ax.set_title("The nave responds to outside weather with a delay of about one day")
ax.legend(ncol=3, fontsize=8)
P.save(fig, "02_lag_correlation.png"); plt.show()
best = pd.DataFrame({"lag of max correlation (h)": lag_corr.idxmax(), "max correlation": lag_corr.max().round(2),
                     "correlation at lag 0": lag_corr.iloc[0].round(2)})
best.index = [P.name(s) for s in best.index]
display(best)
results.save_table(best, "lag_correlation")
results.update(lag_organ_h=int(lag_corr["O1"].idxmax()), lag_basement_h=int(lag_corr["Ke1"].idxmax()),
               lag_attic_h=int(lag_corr["D1"].idxmax()))

# %% [markdown]
# **Decision (correlation and lag):** daily indoor temperatures are almost collinear,
# so they carry little independent information → features use *differences*
# (inside − outside, surface − air, inside − outside absolute humidity) rather than raw
# levels. The attic follows the weather immediately (lag 0, correlation 0.9). The nave
# (organ, walls, north-west art objects) and the basement respond with a delay of about
# **one day (22–25 h)** and only weakly. The upper south art-object sensor reacts
# immediately (lag 0, correlation 0.7) and has a 4–6 °C daily cycle: it sits near the
# south window and is exposed to sun, so its air temperature is not representative of the
# nave; this is reported as a limitation and as a conservation issue (sun on art objects).
# For early warning this means a 24-hour horizon is meaningful: today's weather is
# tomorrow's indoor climate. The model in notebook 04 therefore uses 24-hour means and
# changes of outside conditions.
