"""Derived metrics for the EP all2all benchmark."""
from __future__ import annotations

from typing import List

import torch
import torch.distributed as dist


def dispatch_bytes_per_token(hidden: int, topk: int, dtype_bytes: int) -> int:
    """Algorithmic bytes moved per token during dispatch: each token is
    replicated to its `topk` selected experts."""
    return hidden * topk * dtype_bytes


def achieved_gbps(num_tokens: int, bytes_per_token: int, seconds: float) -> float:
    """Achieved algorithmic bandwidth in GB/s (10^9 bytes)."""
    return (num_tokens * bytes_per_token) / seconds / 1e9


def max_reduce_across_ranks(value: float, group=None) -> float:
    """Return the max of `value` across all ranks (true wall cost is the
    slowest rank)."""
    t = torch.tensor([value], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return float(t.item())


def percentiles(times_s: List[float]) -> dict:
    """p10/p50/p90 in microseconds from a list of per-iter seconds."""
    t = torch.tensor(times_s, dtype=torch.float64)
    return {
        "p10_us": float(torch.quantile(t, 0.10) * 1e6),
        "p50_us": float(torch.quantile(t, 0.50) * 1e6),
        "p90_us": float(torch.quantile(t, 0.90) * 1e6),
    }
