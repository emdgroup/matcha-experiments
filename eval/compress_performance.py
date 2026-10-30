#!/usr/bin/env python3
"""
Convert between the per-run ``performance.json`` files and a single ``performance.csv``.

Each ``eval/<dataset>/<model>/performance.json`` maps a bootstrap sample
(``0_0`` … ``0_999``, plus the aggregated ``mean`` and ``std``) to endpoint → metric → value.
The CSV stores one row per (dataset, model, sample, endpoint) with one column per metric,
keeping the original key order so that the JSON files can be regenerated losslessly.

Usage
-----
python eval/compress_performance.py compress   # performance.json files → performance.csv
python eval/compress_performance.py expand     # performance.csv → performance.json files
"""

import argparse
import csv
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EVAL_DIR = Path(__file__).resolve().parent
CSV_PATH = EVAL_DIR / "performance.csv"
JSON_NAME = "performance.json"

KEY_COLUMNS = ["dataset", "model", "sample", "endpoint"]
METRICS = ["R2", "RMSE", "MAE", "SPEARMAN_R", "Within2Fold", "Within3Fold"]


def compress() -> None:
    paths = sorted(EVAL_DIR.glob(f"*/*/{JSON_NAME}"))
    n_rows = 0
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(KEY_COLUMNS + METRICS)
        for path in paths:
            dataset, model = path.parent.parent.name, path.parent.name
            for sample, endpoints in json.loads(path.read_text()).items():
                for endpoint, scores in endpoints.items():
                    if list(scores) != METRICS:
                        raise ValueError(f"Unexpected metrics in {path}: {list(scores)}")
                    writer.writerow([dataset, model, sample, endpoint] + [repr(scores[m]) for m in METRICS])
                    n_rows += 1
    print(f"  Compressed {len(paths)} files into {CSV_PATH} ({n_rows} rows)")


def expand() -> None:
    runs: dict[tuple[str, str], dict] = {}
    with CSV_PATH.open(newline="") as f:
        for row in csv.DictReader(f):
            run = runs.setdefault((row["dataset"], row["model"]), {})
            run.setdefault(row["sample"], {})[row["endpoint"]] = {m: float(row[m]) for m in METRICS}

    for (dataset, model), performance in runs.items():
        path = EVAL_DIR / dataset / model / JSON_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(performance, indent=4))
    print(f"  Expanded {CSV_PATH} into {len(runs)} {JSON_NAME} files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["compress", "expand"])
    args = parser.parse_args()
    if args.mode == "compress":
        compress()
    else:
        expand()


if __name__ == "__main__":
    main()
