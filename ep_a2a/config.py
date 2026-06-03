"""Configuration for the EP all2all benchmark."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional

BACKENDS = ["deepep", "mooncake", "nixl"]
REGIMES = ["prefill", "decode"]
DTYPE_MODES = ["bf16", "native"]

# Per-regime default token count (tokens generated on each rank).
_DEFAULT_NUM_TOKENS = {"prefill": 4096, "decode": 128}


@dataclass
class BenchConfig:
    backend: str
    regime: str
    dtype_mode: str
    hidden: int = 7168
    num_experts: int = 256
    topk: int = 8
    num_tokens: Optional[int] = None
    routing: str = "balanced"  # "balanced" | "imbalanced"
    warmups: int = 20
    iters: int = 30
    seed: int = 0
    out: str = "result.json"

    def __post_init__(self):
        assert self.backend in BACKENDS, self.backend
        assert self.regime in REGIMES, self.regime
        assert self.dtype_mode in DTYPE_MODES, self.dtype_mode
        assert self.routing in ("balanced", "imbalanced"), self.routing
        if self.num_tokens is None:
            self.num_tokens = _DEFAULT_NUM_TOKENS[self.regime]


def parse_args(argv: Optional[List[str]] = None) -> BenchConfig:
    p = argparse.ArgumentParser(description="EP all2all dispatch/combine benchmark")
    p.add_argument("--backend", required=True, choices=BACKENDS)
    p.add_argument("--regime", required=True, choices=REGIMES)
    p.add_argument("--dtype-mode", required=True, choices=DTYPE_MODES, dest="dtype_mode")
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--num-experts", type=int, default=256, dest="num_experts")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--num-tokens", type=int, default=None, dest="num_tokens")
    p.add_argument("--routing", choices=["balanced", "imbalanced"], default="balanced")
    p.add_argument("--warmups", type=int, default=20)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="result.json")
    a = p.parse_args(argv)
    return BenchConfig(
        backend=a.backend,
        regime=a.regime,
        dtype_mode=a.dtype_mode,
        hidden=a.hidden,
        num_experts=a.num_experts,
        topk=a.topk,
        num_tokens=a.num_tokens,
        routing=a.routing,
        warmups=a.warmups,
        iters=a.iters,
        seed=a.seed,
        out=a.out,
    )
