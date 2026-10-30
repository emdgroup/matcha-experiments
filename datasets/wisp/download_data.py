#!/usr/bin/env python3
"""
Download the experimental LogP dataset used in the WISP explainability benchmark.

The file is fetched from the kerjans/ml-XAI GitHub repository and written to
``datasets/wisp/logP.csv``, the default input of ``explainability/prepare.py``.
"""

import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOGP_URL = "https://raw.githubusercontent.com/kerjans/ml-XAI/69dcffe52be701cf507e9ef15f30d5452d4b091b/datasets/logP.csv"

OUT_PATH = Path(__file__).resolve().parent / "logP.csv"


def main() -> None:
    print(f"  Downloading {LOGP_URL} ...")
    with urllib.request.urlopen(LOGP_URL) as resp:
        OUT_PATH.write_bytes(resp.read())
    print(f"  Saved to {OUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
