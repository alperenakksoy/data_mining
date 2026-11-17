#!/bin/bash
# usage: ./run_nb.sh 01_preprocessing  -> convert py/<name>.py to <name>.ipynb and execute it in place
set -e
cd "$(dirname "$0")"
jupytext --to ipynb --output "$1.ipynb" "py/$1.py" >/dev/null
jupyter nbconvert --to notebook --execute --inplace "$1.ipynb" --ExecutePreprocessor.timeout=1200
