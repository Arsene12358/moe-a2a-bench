"""Single (backend, regime, dtype) benchmark job. Launched once per cell by
run_all.py (or directly via torchrun)."""
from __future__ import annotations

import json

import torch.distributed as dist

from ep_a2a.adapter import correctness_gate, make_phase_fns, run_once
from ep_a2a.bootstrap import build_dispatcher, init_dist_env
from ep_a2a.config import parse_args
from ep_a2a.metrics import (
    achieved_gbps,
    critical_path_series,
    dispatch_bytes_per_token,
    gather_times_matrix,
    per_rank_p50s,
    percentiles,
    rank_spread,
    reduce_rx_counts,
    rx_pairs_per_rank,
    rx_unique_tokens_per_rank,
    skew_max_mean,
    straggler_stability,
)
from ep_a2a.timing import (
    time_fn,
    time_fn_cuda_graph,
    time_phases,
    time_phases_cuda_graph,
)
from ep_a2a.workload import make_workload


def main():
    cfg = parse_args()
    env = init_dist_env(cfg)

    def _write_na(reason: str):
        if env.rank == 0:
            with open(cfg.out, "w") as f:
                json.dump(
                    {
                        "backend": cfg.backend,
                        "regime": cfg.regime,
                        "dtype_mode": cfg.dtype_mode,
                        "routing": cfg.routing,
                        "status": "unavailable",
                        "reason": reason,
                    },
                    f,
                    indent=2,
                )
            print(
                f"skipped: {cfg.backend}/{cfg.regime}/{cfg.dtype_mode} "
                f"unavailable: {reason}",
                flush=True,
            )

    # A backend may be uninstalled (ImportError), built against the wrong torch
    # (ImportError), or not support this regime (NotImplementedError, e.g. NIXL
    # /Mooncake have no normal mode). Mark N/A and keep the sweep alive.
    try:
        dispatcher, num_local_experts = build_dispatcher(cfg, env)
        hidden_states, topk_output = make_workload(
            cfg, rank=env.rank, world_size=env.world_size, device="cuda"
        )
        if cfg.dtype_mode == "bf16":
            correctness_gate(dispatcher, hidden_states, topk_output)
            if env.rank == 0:
                print("correctness gate: OK", flush=True)
        else:
            run_once(dispatcher, hidden_states, topk_output)  # probe
    except (ImportError, NotImplementedError) as e:
        _write_na(f"{type(e).__name__}: {str(e)[:160]}")
        dist.barrier()
        dist.destroy_process_group()
        return

    dist.barrier()
    _time_one = time_fn_cuda_graph if cfg.cuda_graph else time_fn
    _time_split = time_phases_cuda_graph if cfg.cuda_graph else time_phases
    if cfg.split_phases:
        d_fn, c_fn = make_phase_fns(dispatcher, hidden_states, topk_output)
        phase_times = _time_split(
            [("dispatch", d_fn), ("combine", c_fn)],
            warmups=cfg.warmups,
            iters=cfg.iters,
        )
        times = [
            d + c for d, c in zip(phase_times["dispatch"], phase_times["combine"])
        ]
    else:
        phase_times = None
        times = _time_one(
            lambda: run_once(dispatcher, hidden_states, topk_output),
            warmups=cfg.warmups,
            iters=cfg.iters,
        )

    # rank x iter matrix: critical path is a per-iteration max across ranks,
    # computed BEFORE percentiles (max-of-p50s underestimates under jitter).
    matrix = gather_times_matrix(times, env.world_size)
    crit = percentiles(critical_path_series(matrix).tolist())
    rank_p50_us = [v * 1e6 for v in per_rank_p50s(matrix)]
    spread = rank_spread(rank_p50_us)
    stability = straggler_stability(matrix)

    # Achieved routed load per destination rank (intended distribution is set
    # by --routing; this measures what this world size / placement realized).
    rx_pairs = reduce_rx_counts(
        rx_pairs_per_rank(topk_output.topk_ids, cfg.num_experts, env.world_size)
    )
    rx_unique = reduce_rx_counts(
        rx_unique_tokens_per_rank(
            topk_output.topk_ids, cfg.num_experts, env.world_size
        )
    )
    rx_skew = skew_max_mean(rx_pairs)

    dtype_bytes = 2 if cfg.dtype_mode == "bf16" else 1
    bpt = dispatch_bytes_per_token(cfg.hidden, cfg.topk, dtype_bytes)
    crit_p50_s = crit["p50_us"] / 1e6
    gbps = achieved_gbps(cfg.num_tokens, bpt, crit_p50_s)
    # The lane that saturates first under skew: the hottest rank's receive.
    hot_rank_rx_gbps = (
        int(rx_pairs.max()) * cfg.hidden * dtype_bytes / crit_p50_s / 1e9
    )

    result = {
        "backend": cfg.backend,
        "regime": cfg.regime,
        "dtype_mode": cfg.dtype_mode,
        "timing": "cuda_graph" if cfg.cuda_graph else "eager",
        "routing": cfg.routing,
        "skew": cfg.skew if cfg.routing == "zipf" else None,
        "hot_placement": cfg.hot_placement if cfg.routing == "zipf" else None,
        "trace_path": cfg.trace_path,
        "world_size": env.world_size,
        "num_tokens": cfg.num_tokens,
        "hidden": cfg.hidden,
        "num_experts": cfg.num_experts,
        "topk": cfg.topk,
        # headline: per-iteration critical path (max across ranks, then p50)
        "roundtrip_p50_us": crit["p50_us"],
        "critical_path_p10_us": crit["p10_us"],
        "critical_path_p50_us": crit["p50_us"],
        "critical_path_p90_us": crit["p90_us"],
        "per_rank_p50_us": rank_p50_us,
        "rank_spread": spread,
        "straggler_stability": stability,
        "rx_pairs_per_rank": [int(v) for v in rx_pairs],
        "rx_unique_tokens_per_rank": [int(v) for v in rx_unique],
        "rx_skew_max_mean": rx_skew,
        "achieved_gbps": gbps,
        "hot_rank_rx_gbps": hot_rank_rx_gbps,
    }
    if phase_times is not None:
        for name in ("dispatch", "combine"):
            m = gather_times_matrix(phase_times[name], env.world_size)
            pc = percentiles(critical_path_series(m).tolist())
            result[f"{name}_p50_us"] = pc["p50_us"]
            result[f"{name}_per_rank_p50_us"] = [
                v * 1e6 for v in per_rank_p50s(m)
            ]

    if env.rank == 0:
        with open(cfg.out, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
