"""Synthetic inputs for the EP all2all benchmark.

Produces a hidden_states tensor and a StandardTopKOutput with either a
balanced (uniform) or imbalanced (hotspot) expert assignment. a2a cost is
sensitive to load skew, so both distributions are supported.
"""
from __future__ import annotations

import torch

from sglang.srt.layers.moe.topk import StandardTopKOutput

from ep_a2a.config import BenchConfig


def make_workload(cfg: BenchConfig, rank: int, device: str = "cuda"):
    g = torch.Generator(device=device).manual_seed(cfg.seed + rank)
    hidden_states = torch.randn(
        cfg.num_tokens, cfg.hidden, dtype=torch.bfloat16, device=device, generator=g
    )

    if cfg.routing == "balanced":
        # Uniform random scores over all experts.
        scores = torch.rand(
            cfg.num_tokens, cfg.num_experts, device=device, generator=g
        )
    else:
        # Hotspot: bias a small set of experts so load is skewed.
        scores = torch.rand(
            cfg.num_tokens, cfg.num_experts, device=device, generator=g
        )
        hot = max(1, cfg.num_experts // 8)
        scores[:, :hot] += 3.0

    topk_weights, topk_ids = torch.topk(scores, cfg.topk, dim=-1)
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int64)
    # router_logits is unused by the dispatch path; pass the scores as a stand-in.
    return hidden_states, StandardTopKOutput(topk_weights, topk_ids, scores)
