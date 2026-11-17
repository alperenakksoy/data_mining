# Microclimate, mold risk and organ stability in a historic church

Data mining project (winter semester 2026/27). A diocese installed temperature and humidity
sensors in and around a church. We clean the sensor time series, describe the building's
microclimate, assess the **mold risk** per location with a transparent rule based on the
Fraunhofer IBP guideline (LIM isopleths, dew point margin, duration), analyse **temperature
stability at the organ**, and turn the results into a client report with prioritised
recommendations.

**Deliverables**

| What | Where |
|---|---|
| Client report with technical appendix (PDF) | [`report/report.pdf`](report/report.pdf) |
| Executed notebooks 01–06 | [`notebook/`](notebook) |
| Reusable pipeline code | [`src/`](src) |
| Figures and result tables used in the report | `report/figures/`, `report/tables/` |

## Reproduce everything

Requirements: Python 3.13 (versions in `requirements.txt`). Building the PDF also needs
`pandoc` (3.x) and Node.js with the `playwright` package (headless Chromium).

```bash
pip install -r requirements.txt
```

```bash
python -m src.run_all
```

This single command:

1. runs the preprocessing pipeline (`python -m src.load_data`) and writes
   `data/processed/clean.parquet` plus a change log,
2. executes notebooks `01_preprocessing` … `06_story` top to bottom and saves their outputs,
3. builds `report/report.pdf` (notebook 06 calls `python -m src.report`).

Single steps: `python -m src.load_data` (cleaning only), `python -m src.report` (PDF only, from
existing tables), `python -m src.run_all --only 05_organ` (one notebook). Each notebook also runs
with "Restart Kernel and Run All". Random seeds are fixed (42).

## Repository layout

```
data/raw/            original CSV (never modified)
data/processed/      generated: clean.parquet, features.parquet, risk.parquet, preprocessing_log.csv
notebook/            01_preprocessing … 06_story (.ipynb, executed)
notebook/py/         plain-text (jupytext percent format) sources of the notebooks, for readable diffs
src/config.py        paths, sensor dictionary, every threshold in one place
src/load_data.py     cleaning pipeline (+ load_processed())
src/physics.py       Magnus formula, surface relative humidity, LIM curves
src/features.py      dew point margin, LIM excess, duration and cross-sensor features
src/risk.py          rule-based risk index, summaries, common time windows
src/model.py         optional early-warning decision tree with time split and baselines
src/organ.py         time-based one-hour organ events and episodes
src/plotting.py      plotting style, floor plan, calendar heatmap
src/report.py        template -> Markdown -> HTML -> PDF
src/run_all.py       one-command reproduction
report/              report template, CSS, figures, tables, PDF
```

## Notebooks

| Notebook | Content |
|---|---|
| `01_preprocessing` | schema, sensor dictionary, time zone handling, every cleaning step with evidence, availability heatmap |
| `02_eda` | missing data and gap lengths, interpolation hold-out test and sensitivity (limits 0/2/3/6), sanity checks, sun on sensors, seasonal / daily / weekday patterns, indoor vs outdoor, correlation and 0–48 h lagged response |
| `03_features` | dew point margin (0/2/3/4 °C), LIM class I/II excess, surface relative humidity, rolling 24 h / 7 d / 30 d exposure, episodes, ventilation potential |
| `04_risk` | rule-based risk index, monthly profile, episodes, fair comparison on shared time windows, 36-configuration sensitivity, physical plausibility, early-warning tree vs persistence, error analysis, limitations |
| `05_organ` | time-based one-hour events vs row shift, raw vs clean vs no interpolation, episodes, per year / heating season, month / hour / weekday, link to heating |
| `06_story` | client figures (floor plan, risk calendar, organ) and PDF build |

## Main results (details in the report)

* **Mold risk ranking** (fair comparison on shared time): basement storage area > art objects in
  the north-west corner > organ ≈ north exterior wall ≈ art objects at the south wall > south wall
  > attic > east wall. The top two and the bottom location are stable in all 36 threshold settings.
* Basement and cold north-west corner are critical in **summer**, the whole nave in
  **October–November**; winter and spring are calm.
* **Organ**: 74 readings / 23 episodes with a change > 1 °C per hour, all warm-ups at heating starts
  in the heating season, mostly on Fridays and Saturdays. About a third of the raw "events" (the
  brief's preliminary 111) were sensor artefacts (installation day, one-reading glitches).
* The early-warning model does not reliably beat simple persistence; the transparent rule is
  recommended for alerts.

## Data notes

* Raw file: semicolon separated, decimal comma, missing code `-255`, UTC timestamps every 20 minutes.
* Cleaning decisions (all logged and flagged, nothing deleted silently): humidity > 100 % at the
  outside sensor clipped; installation day removed; organ air sensor relocation excluded; 14 joint
  temperature/humidity glitches at the organ removed; 2 h warm-up after outages removed; evening sun
  on the outside sensor flagged; gaps ≤ 1 h interpolated.
* No wall sensor covers winter 2024/25; this limits wall comparisons (see report, section 6).

## Working with Git

Development happened on feature branches (`feature/preprocessing`, `feature/eda`,
`feature/features`, `feature/cleaning-artefacts`, `feature/risk`, `feature/organ`,
`feature/story-report`) merged with `--no-ff` so the history shows each work package.
Generated data in `data/processed/` is not committed; it is rebuilt by the pipeline.
