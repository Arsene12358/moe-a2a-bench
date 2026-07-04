"""Synthetic and replayed inputs for the EP all2all benchmark.

Routing modes (a2a cost is sensitive to load skew, so several are supported):

- "balanced":  uniform k-subsets (iid scores + topk; the original behavior).
- "hotspot":   legacy step-function skew — the first num_experts//8 experts
               deterministically win every token ("imbalanced" is an alias).
               Kept as the worst-case corner; NOT calibrated.
- "zipf":      calibrated skew. Expert popularity p_i ∝ 1/(i+1)^skew, sampled
               per token WITHOUT replacement via Gumbel-top-k. skew=0 is
               balanced; production MoE routing typically lands ~0.5-1.5.
               --hot-placement decides whether the popular experts concentrate
               on the first ranks ("contiguous") or spread across all ranks
               ("scattered", default; matches real models, where hot experts
               are not adjacent ids).
- "trace":     replay real routing from --trace-path (.npz with topk_ids
               [N, topk] int64 and optional topk_weights [N, topk] float32;
               e.g. converted from sglang's expert_distribution_recorder).
               Rank r deterministically takes rows [r*num_tokens, ...) with
               wraparound, so all N rows are exercised across ranks.

The intended-vs-achieved distinction matters: these modes set the *intended*
distribution; the report's rx_pairs_per_rank / rx_skew_max_mean measure what
was *achieved* on this world size and placement.
"""
from __future__ import annotations

import numpy as np
import torch

from ep_a2a.config import BenchConfig


def zipf_expert_probs(
    num_experts: int,
    skew: float,
    world_size: int,
    hot_placement: str = "scattered",
    device: str = "cpu",
) -> torch.Tensor:
    """Expert popularity vector p [num_experts], sum 1.

    Popularity rank j gets mass ∝ 1/(j+1)^skew. With "contiguous" placement
    popularity rank j IS expert id j (hot experts pile onto the first ranks
    under the dispatcher's contiguous expert->rank map — the worst case).
    With "scattered", experts are placed load-aware greedily (LPT: next
    hottest expert goes to the least-loaded rank with a free slot) — the
    best-case static placement, an EPLB-like envelope. Real unoptimized
    placements fall between the two. Note a heavy head bounds how even any
    placement can be: whoever owns the #1 expert carries its full mass."""
    assert num_experts % world_size == 0
    j = torch.arange(num_experts, dtype=torch.float64, device=device)
    mass = (j + 1.0) ** (-skew)
    p = torch.empty(num_experts, dtype=torch.float64, device=device)
    if hot_placement == "contiguous":
        p[:] = mass
    elif hot_placement == "scattered":
        local = num_experts // world_size
        load = [0.0] * world_size
        slots = [0] * world_size
        for jj in range(num_experts):  # mass is already sorted descending
            r = min(
                (r for r in range(world_size) if slots[r] < local),
                key=lambda r: load[r],
            )
            p[r * local + slots[r]] = float(mass[jj])
            load[r] += float(mass[jj])
            slots[r] += 1
    else:
        raise ValueError(f"unknown hot_placement: {hot_placement}")
    return (p / p.sum()).to(torch.float32)


def gumbel_topk_ids(
    probs: torch.Tensor,
    num_tokens: int,
    topk: int,
    generator: torch.Generator,
):
    """Sample topk expert ids per token WITHOUT replacement from `probs` via
    the Gumbel-top-k trick. Returns (scores [n, E], topk_ids [n, topk]).
    With uniform probs this reduces exactly to uniform k-subsets."""
    device = probs.device
    u = torch.rand(
        (num_tokens, probs.numel()), device=device, generator=generator
    ).clamp_(min=1e-10)
    gumbel = -torch.log(-torch.log(u))
    scores = torch.log(probs).unsqueeze(0) + gumbel
    _, topk_ids = torch.topk(scores, topk, dim=-1)
    return scores, topk_ids


def load_trace_slice(
    trace_path: str,
    rank: int,
    num_tokens: int,
    topk: int,
    num_experts: int,
):
    """Load real routing and take this rank's deterministic slice.

    Returns (topk_ids [num_tokens, topk] int64,
             topk_weights [num_tokens, topk] float32)."""
    data = np.load(trace_path)
    if "topk_ids" not in data:
        raise ValueError(f"{trace_path}: missing 'topk_ids' array")
    ids = torch.from_numpy(np.ascontiguousarray(data["topk_ids"])).long()
    if ids.ndim != 2:
        raise ValueError(f"{trace_path}: topk_ids must be 2-D, got {tuple(ids.shape)}")
    if ids.shape[1] != topk:
        raise ValueError(
            f"{trace_path}: trace topk={ids.shape[1]} != configured topk={topk}; "
            "set --topk to match the trace"
        )
    if int(ids.min()) < 0 or int(ids.max()) >= num_experts:
        raise ValueError(
            f"{trace_path}: expert ids [{int(ids.min())}, {int(ids.max())}] out of "
            f"range for num_experts={num_experts}; set --num-experts to match"
        )
    n = ids.shape[0]
    rows = (torch.arange(num_tokens) + rank * num_tokens) % n
    ids = ids[rows]

    if "topk_weights" in data:
        w = torch.from_numpy(np.ascontiguousarray(data["topk_weights"])).float()[rows]
        if w.shape != ids.shape:
            raise ValueError(f"{trace_path}: topk_weights shape != topk_ids shape")
    else:
        w = torch.full_like(ids, 1.0 / topk, dtype=torch.float32)
    return ids, w


def make_workload(cfg: BenchConfig, rank: int, world_size: int, device: str = "cuda"):
    """Build (hidden_states, StandardTopKOutput) for one rank."""
    # sglang import stays inside the function so the pure helpers above remain
    # unit-testable without sglang installed.
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    g = torch.Generator(device=device).manual_seed(cfg.seed + rank)
    hidden_states = torch.randn(
        cfg.num_tokens, cfg.hidden, dtype=torch.bfloat16, device=device, generator=g
    )

    if cfg.routing == "balanced":
        scores = torch.rand(
            cfg.num_tokens, cfg.num_experts, device=device, generator=g
        )
        topk_weights, topk_ids = torch.topk(scores, cfg.topk, dim=-1)
    elif cfg.routing == "hotspot":
        # Legacy worst case: a small deterministic hot set wins every token.
        scores = torch.rand(
            cfg.num_tokens, cfg.num_experts, device=device, generator=g
        )
        hot = max(1, cfg.num_experts // 8)
        scores[:, :hot] += 3.0
        topk_weights, topk_ids = torch.topk(scores, cfg.topk, dim=-1)
    elif cfg.routing == "zipf":
        probs = zipf_expert_probs(
            cfg.num_experts, cfg.skew, world_size, cfg.hot_placement, device=device
        )
        scores, topk_ids = gumbel_topk_ids(probs, cfg.num_tokens, cfg.topk, g)
        # positive, sum-1 weights per token (production router shape)
        topk_weights = torch.softmax(torch.gather(scores, 1, topk_ids), dim=-1)
    elif cfg.routing == "trace":
        topk_ids, topk_weights = load_trace_slice(
            cfg.trace_path, rank, cfg.num_tokens, cfg.topk, cfg.num_experts
        )
        topk_ids = topk_ids.to(device)
        topk_weights = topk_weights.to(device)
        # router_logits stand-in (unused by the dispatch path).
        scores = torch.zeros(
            cfg.num_tokens, cfg.num_experts, device=device
        ).scatter_(1, topk_ids, topk_weights)
    else:  # unreachable: config validates
        raise ValueError(f"unknown routing: {cfg.routing}")

    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int64)
    return hidden_states, StandardTopKOutput(topk_weights, topk_ids, scores)
