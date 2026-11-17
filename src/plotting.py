"""Shared plotting style and client-facing figures (floor plan, calendar heatmap).

Colours: a fixed categorical order (colour-blind validated) for locations and
reserved status colours green / yellow / red for risk levels, always shown
together with a text label.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from . import config as C

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#52514e"]
LOCATION_COLORS = {s: SERIES[i] for i, s in enumerate(["O1", "AWn", "AWs", "AWo", "K1", "K2", "D1", "Ke1", "AU"])}
STATUS = {"low": "#0ca30c", "medium": "#fab219", "high": "#d03b3b", "unknown": "#b8b7b1"}
LEVEL_NAMES = {0: "low", 1: "medium", 2: "high"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ_RED = LinearSegmentedColormap.from_list("seq_red", ["#fbeeee", "#efb3ab", "#e07468", "#c8302b", "#7d1814"])
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ec8a7f", "#b8302b"])


def setup() -> None:
    """Apply a calm, readable default style to all matplotlib figures."""
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": INK2, "axes.edgecolor": "#b8b7b1", "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 1.6, "legend.frameon": False, "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def name(code: str, short: bool = True) -> str:
    return C.client_name(code, short=short)


def save(fig, filename: str) -> None:
    """Save a figure to ``report/figures`` (PNG)."""
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG_DIR / filename)


def risk_color(level) -> str:
    if level is None or (isinstance(level, float) and np.isnan(level)):
        return STATUS["unknown"]
    return STATUS[LEVEL_NAMES[int(level)]]


# --------------------------------------------------------------------------
# Floor plan
# --------------------------------------------------------------------------
# Approximate sensor positions on the schematic of the brief (x to the east, y to the north)
PLAN_POS = {"AU": (0.9, 5.0), "O1": (2.3, 5.0), "K1": (2.9, 7.5), "AWn": (6.3, 7.5),
            "K2": (6.0, 2.5), "AWs": (6.6, 2.5), "AWo": (13.3, 5.0)}


def floor_plan(levels: dict, title: str, ax=None, labels: dict | None = None):
    """Draw the church schematic with sensor markers coloured by risk level.

    ``levels``: location code -> 0/1/2 or NaN (no data). Attic and basement are
    shown as separate boxes because they are not on the ground plan.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5.2))
    else:
        fig = ax.figure
    ax.set_aspect("equal")
    ax.axis("off")
    body = "#dbe6f3"
    ax.add_patch(Rectangle((2.0, 2.5), 10.0, 5.0, fc=body, ec="#7d93ad", lw=1.2))   # nave
    ax.add_patch(Rectangle((1.2, 4.2), 0.8, 1.6, fc=body, ec="#7d93ad", lw=1.2))    # west tower
    ax.add_patch(Circle((1.6, 6.1), 0.45, fc=body, ec="#7d93ad", lw=1.2))
    ax.add_patch(Circle((1.6, 3.9), 0.45, fc=body, ec="#7d93ad", lw=1.2))
    ax.add_patch(Rectangle((12.0, 5.8), 1.6, 1.7, fc=body, ec="#7d93ad", lw=1.2))   # east annexes
    ax.add_patch(Rectangle((12.0, 2.5), 1.6, 1.7, fc=body, ec="#7d93ad", lw=1.2))
    ax.add_patch(Rectangle((12.0, 4.2), 1.0, 1.6, fc=body, ec="#7d93ad", lw=1.2))
    ax.text(7.0, 5.0, "Nave", ha="center", va="center", color="#7d93ad", fontsize=13, style="italic")
    ax.annotate("N", xy=(0.3, 3.3), xytext=(0.3, 1.6), ha="center", fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.5))
    offsets = {"AU": (0, -1.0), "O1": (0.25, -0.9), "K1": (0, 0.75), "AWn": (0, 0.75),
               "K2": (-0.9, -0.95), "AWs": (0.9, -0.95), "AWo": (0.5, -1.5)}
    for s, (x, y) in PLAN_POS.items():
        lvl = levels.get(s, np.nan)
        ax.add_patch(Circle((x, y), 0.32, fc=risk_color(lvl), ec="white", lw=2, zorder=5))
        dx, dy = offsets[s]
        txt = labels.get(s, name(s)) if labels else name(s)
        ax.text(x + dx, y + dy, txt, ha="center", va="center", fontsize=8.5, color=INK, zorder=6)
    # attic and basement boxes
    for i, s in enumerate(["D1", "Ke1"]):
        bx, by = 14.6, 6.0 - i * 2.6
        ax.add_patch(FancyBboxPatch((bx, by), 2.6, 1.6, boxstyle="round,pad=0.05", fc="#f4f3f0", ec="#b8b7b1"))
        lvl = levels.get(s, np.nan)
        ax.add_patch(Circle((bx + 0.5, by + 0.8), 0.32, fc=risk_color(lvl), ec="white", lw=2, zorder=5))
        txt = labels.get(s, name(s)) if labels else name(s)
        ax.text(bx + 1.0, by + 0.8, txt, va="center", fontsize=8.5, color=INK)
    ax.text(14.6, 7.9, "Not on the ground plan", fontsize=8, color=INK2)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=10, mfc=STATUS[k], mec="white", label=lab)
               for k, lab in [("high", "High risk"), ("medium", "Medium risk"), ("low", "Low risk"),
                              ("unknown", "No data")]]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=9)
    ax.set_xlim(-0.2, 17.5)
    ax.set_ylim(0.6, 8.9)
    ax.set_title(title)
    return fig, ax


def calendar_heatmap(values: pd.DataFrame, title: str, cbar_label: str, ax=None, vmax: float = 1.0,
                     cmap=SEQ_RED, fmt_pct: bool = True):
    """Location x period heatmap; grey = no data. ``values`` rows = periods, columns = locations."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 0.45 * values.shape[1] + 1.6))
    else:
        fig = ax.figure
    data = values.T
    m = np.ma.masked_invalid(data.to_numpy(dtype=float))
    cm = cmap.copy()
    cm.set_bad("#e4e3df")
    im = ax.imshow(m, aspect="auto", cmap=cm, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(data.shape[0]), [name(s) if s in C.SENSORS else s for s in data.index])
    cols = list(data.columns)
    step = max(1, len(cols) // 24)
    ax.set_xticks(range(0, len(cols), step), [str(c) for c in cols[::step]], rotation=90)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(cbar_label)
    if fmt_pct:
        cb.ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_title(title)
    return fig, ax
