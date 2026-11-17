"""Build the PDF report from ``report/report_template.md``.

The template contains placeholders that are filled from ``report/tables``:

* ``{{key}}`` or ``{{key.sub|fmt}}`` -> value from ``results.json`` with an optional
  format (``pct``, ``pct1``, ``int``, ``f1``, ``f2``, ``k`` for thousands separator)
* ``{{table:name}}`` or ``{{table:name|pct=col1,col2}}`` -> CSV ``report/tables/name.csv``
  rendered as a Markdown table

Pipeline: template -> Markdown -> HTML (pandoc) -> PDF (headless Chromium via the
Node.js ``playwright`` package, script ``report/make_pdf.js``).

Run with ``python -m src.report``.
"""
from __future__ import annotations

import re
import subprocess

import pandas as pd

from . import config as C
from .results import load

TEMPLATE = C.REPORT_DIR / "report_template.md"
CSS = C.REPORT_DIR / "report.css"
BUILD = C.REPORT_DIR / "_build"
PDF = C.REPORT_DIR / "report.pdf"


def _fmt(v, f: str | None) -> str:
    if v is None:
        return "n/a"
    if f == "pct":
        return f"{v:.0%}"
    if f == "pct1":
        return f"{v:.1%}"
    if f == "int":
        return f"{int(round(v)):,}"
    if f == "f1":
        return f"{v:.1f}"
    if f == "f2":
        return f"{v:.2f}"
    if isinstance(v, float):
        return f"{v:,.2f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _lookup(data: dict, key: str):
    cur = data
    for part in key.split("."):
        cur = cur[part]
    return cur


def md_table(df: pd.DataFrame, pct_cols=(), digits: int = 2) -> str:
    """Render a DataFrame as a pipe table (no extra dependency)."""
    df = df.copy()
    for c in df.columns:
        if c in pct_cols:
            df[c] = df[c].map(lambda x: "–" if pd.isna(x) else f"{x:.0%}")
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda x: "–" if pd.isna(x) else (f"{x:,.0f}" if abs(x) >= 100 else f"{x:.{digits}f}"))
    df = df.fillna("–").astype(str)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("------" if i == 0 else "---:" for i in range(len(df.columns))) + "|"
    rows = ["| " + " | ".join(r) + " |" for r in df.to_numpy()]
    return "\n".join([head, sep, *rows])


def render(template: str, data: dict) -> str:
    def table(m):
        spec = m.group(1)
        name, _, opts = spec.partition("|")
        df = pd.read_csv(C.TABLE_DIR / f"{name.strip()}.csv")
        pct = []
        for opt in filter(None, opts.split(";")):
            k, _, v = opt.partition("=")
            if k.strip() == "pct":
                pct = [x.strip() for x in v.split(",")]
            if k.strip() == "cols":
                df = df[[x.strip() for x in v.split(",")]]
        return md_table(df, pct)

    out = re.sub(r"\{\{table:([^}]+)\}\}", table, template)
    out = re.sub(r"\{\{([A-Za-z0-9_.\-/ ]+?)(?:\|([a-z0-9]+))?\}\}",
                 lambda m: _fmt(_lookup(data, m.group(1).strip()), m.group(2)), out)
    return out


def build() -> "pathlib.Path":
    """Render the template, convert to HTML and print to PDF. Returns the PDF path."""
    BUILD.mkdir(parents=True, exist_ok=True)
    md = render(TEMPLATE.read_text(encoding="utf-8"), load())
    md_file = BUILD / "report.md"
    md_file.write_text(md, encoding="utf-8")
    html_file = BUILD / "report.html"
    subprocess.run(["pandoc", str(md_file), "-o", str(html_file), "--standalone", "--embed-resources",
                    f"--resource-path={C.REPORT_DIR}", f"--css={CSS}", "--mathml",
                    "--metadata", "lang=en"], check=True)
    subprocess.run(["node", str(C.REPORT_DIR / "make_pdf.js"), str(html_file), str(PDF)], check=True)
    return PDF


if __name__ == "__main__":
    print(build())
