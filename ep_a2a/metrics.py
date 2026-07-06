"""Derived metrics for the EP all2all benchmark.

Pure helpers operate on CPU tensors/lists so they are unit-testable without
GPUs or an initialized process group; the dist collectives live at the bottom
and are only called from bench.py at runtime.
"""
from __future__ import annotations

from typing import List

import torch
import torch.distributed as dist


def dispatch_bytes_per_token(hidden: int, topk: int, dtype_bytes: int) -> int:
    """Algorithmic bytes moved per token during dispatch: each token is
    replicated to its `topk` selected experts."""
    return hidden * topk * dtype_bytes


def dispatch_wire_bytes_per_token(
    hidden: int, topk: int, payload_bytes: int, scale_bytes_per_elem: float
) -> float:
    """Actual wire bytes per token during dispatch, from the probed dispatch
    output format (payload element size + amortized quantization scales)."""
    return hidden * topk * (payload_bytes + scale_bytes_per_elem)


def achieved_gbps(num_tokens: int, bytes_per_token: int, seconds: float) -> float:
    """Achieved algorithmic bandwidth in GB/s (10^9 bytes)."""
    return (num_tokens * bytes_per_token) / seconds / 1e9


def percentiles(times_s: List[float]) -> dict:
    """p10/p50/p90 in microseconds from a list of per-iter seconds."""
    t = torch.tensor(times_s, dtype=torch.float64)
    return {
        "p10_us": float(torch.quantile(t, 0.10) * 1e6),
        "p50_us": float(torch.quantile(t, 0.50) * 1e6),
        "p90_us": float(torch.quantile(t, 0.90) * 1e6),
    }


# ---------------------------------------------------------------------------
# rank x iter matrix analysis (pure)
# ---------------------------------------------------------------------------


def critical_path_series(times_matrix: torch.Tensor) -> torch.Tensor:
    """Per-iteration wall time: max across ranks, computed within each
    iteration BEFORE any percentile (max-of-p50s underestimates whenever the
    slowest rank rotates between iterations). times_matrix is
    [world_size, iters]; returns [iters]."""
    return times_matrix.max(dim=0).values


def per_rank_p50s(times_matrix: torch.Tensor) -> List[float]:
    """Median time per rank, in seconds (interpolating quantile, consistent
    with percentiles(); torch.median would take the lower middle element)."""
    return [float(v) for v in torch.quantile(times_matrix.double(), 0.5, dim=1)]


def rank_spread(per_rank_p50: List[float]) -> float:
    """max/min of per-rank medians. 1.0 = perfectly even; grows with load
    imbalance. This is the load-balance cost signal."""
    lo = min(per_rank_p50)
    return float(max(per_rank_p50) / lo) if lo > 0 else float("inf")


def straggler_stability(times_matrix: torch.Tensor) -> float:
    """Fraction of iterations whose slowest rank is the modal slowest rank.

    ~1.0 -> statically hot rank (load imbalance / placement problem);
    ~1/world_size -> rotating stragglers (congestion / jitter).
    """
    slowest = times_matrix.argmax(dim=0)
    modal = int(torch.mode(slowest).values)
    return float((slowest == modal).float().mean())


# ---------------------------------------------------------------------------
# routed-load characterization (pure)
# ---------------------------------------------------------------------------


def expert_to_rank(
    topk_ids: torch.Tensor, num_experts: int, world_size: int
) -> torch.Tensor:
    """Map expert ids to owning rank under the contiguous placement used by
    build_dispatcher (num_local_experts = num_experts // world_size)."""
    assert num_experts % world_size == 0
    return topk_ids // (num_experts // world_size)


def rx_pairs_per_rank(
    topk_ids: torch.Tensor, num_experts: int, world_size: int
) -> torch.Tensor:
    """This rank's contribution to each destination rank's received
    (token, expert) pair count. Pair count is what sizes both the receive
    traffic and the expert GEMM. Returns int64 [world_size]."""
    dest = expert_to_rank(topk_ids, num_experts, world_size)
    return torch.bincount(dest.flatten(), minlength=world_size)


def rx_unique_tokens_per_rank(
    topk_ids: torch.Tensor, num_experts: int, world_size: int
) -> torch.Tensor:
    """This rank's contribution to each destination rank's received unique
    token count (a token selecting two experts on the same rank counts once).
    Relevant for backends that deduplicate per destination. int64 [world_size]."""
    dest = expert_to_rank(topk_ids, num_experts, world_size)  # [tokens, topk]
    onehot = torch.zeros(
        (dest.shape[0], world_size), dtype=torch.bool, device=dest.device
    )
    onehot.scatter_(1, dest, True)
    return onehot.sum(dim=0).to(torch.int64)


def skew_max_mean(counts: torch.Tensor) -> float:
    """max/mean of a per-rank load vector. 1.0 = perfectly balanced."""
    m = float(counts.to(torch.float64).mean())
    return float(counts.max()) / m if m > 0 else float("inf")


# ---------------------------------------------------------------------------
# collectives (thin, runtime-only)
# ---------------------------------------------------------------------------


def gather_times_matrix(
    times_s: List[float], world_size: int, group=None
) -> torch.Tensor:
    """all_gather each rank's per-iter times; returns [world_size, iters] on
    CPU. Called after the timed loop ends, so it perturbs nothing."""
    local = torch.tensor(times_s, dtype=torch.float64, device="cuda")
    bufs = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(bufs, local, group=group)
    return torch.stack(bufs).cpu()


def reduce_rx_counts(local_counts: torch.Tensor, group=None) -> torch.Tensor:
    """Sum per-destination counts over all source ranks."""
    t = local_counts.to(torch.int64).cuda()
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
    return t.cpu()


def max_reduce_across_ranks(value: float, group=None) -> float:
    """Return the max of `value` across all ranks (kept for compatibility;
    bench.py now uses the gathered rank x iter matrix instead)."""
    t = torch.tensor([value], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return float(t.item())
