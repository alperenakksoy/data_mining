"""Reproduce everything with one command from the repository root::

    python -m src.run_all

1. preprocessing pipeline (``src/load_data.py``) -> ``data/processed/clean.parquet``
2. execute notebooks 01-06 top to bottom (outputs are saved in the notebooks)
3. notebook 06 builds ``report/report.pdf``

Options: ``--only 04_risk`` runs a single notebook.
"""
from __future__ import annotations

import argparse
import time

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from . import config as C
from .load_data import main as preprocess

NOTEBOOKS = ["01_preprocessing", "02_eda", "03_features", "04_risk", "05_organ", "06_story"]


def run_notebook(name: str) -> None:
    path = C.ROOT / "notebook" / f"{name}.ipynb"
    nb = nbformat.read(path, as_version=4)
    t0 = time.time()
    ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(path.parent)}})
    nbformat.write(nb, path)
    print(f"  {name}: ok ({time.time() - t0:.0f} s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None, help="notebook names to run")
    args = ap.parse_args()
    print("1/2 preprocessing")
    preprocess()
    print("2/2 notebooks")
    for name in args.only or NOTEBOOKS:
        run_notebook(name)


if __name__ == "__main__":
    main()
