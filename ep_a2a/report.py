"""Aggregate per-job result JSONs into a comparison table."""
from __future__ import annotations

import argparse
import glob
import json
from typing import List

_COLUMNS = [
    "backend",
    "regime",
    "dtype_mode",
    "routing",
    "roundtrip_p50_us",
    "rank_spread",
    "rx_skew_max_mean",
    "achieved_gbps",
]
_NUMERIC = ("roundtrip_p50_us", "rank_spread", "rx_skew_max_mean", "achieved_gbps")


def _cell(r: dict, key: str) -> str:
    if r.get("status") == "unavailable":
        return "N/A" if key in _NUMERIC else str(r.get(key, ""))
    val = r.get(key, "")
    if isinstance(val, float):
        return f"{val:.2f}" if key in ("rank_spread", "rx_skew_max_mean") else f"{val:.1f}"
    return str(val)


def format_table(results: List[dict]) -> str:
    rows = [[_cell(r, c) for c in _COLUMNS] for r in results]
    widths = [
        max(len(c), *(len(row[i]) for row in rows)) for i, c in enumerate(_COLUMNS)
    ]

    def fmt(cells):
        return " | ".join(c.ljust(w) for c, w in zip(cells, widths))

    sep = "-+-".join("-" * w for w in widths)
    return "\n".join([fmt(_COLUMNS), sep, *(fmt(r) for r in rows)])


def load_results(pattern: str) -> List[dict]:
    out = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            out.append(json.load(f))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="result_*.json")
    a = p.parse_args()
    results = load_results(a.glob)
    print(format_table(results))


if __name__ == "__main__":
    main()
