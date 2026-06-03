"""Single (backend, regime, dtype) benchmark job. Launched once per cell by
run_all.py (or directly via torchrun)."""
from __future__ import annotations

import json

import torch.distributed as dist

from ep_a2a.adapter import correctness_gate, run_once
from ep_a2a.bootstrap import build_dispatcher, init_dist_env
from ep_a2a.config import parse_args
from ep_a2a.metrics import (
    achieved_gbps,
    dispatch_bytes_per_token,
    max_reduce_across_ranks,
    percentiles,
)
from ep_a2a.timing import time_fn
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
        hidden_states, topk_output = make_workload(cfg, rank=env.rank, device="cuda")
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
    times = time_fn(
        lambda: run_once(dispatcher, hidden_states, topk_output),
        warmups=cfg.warmups,
        iters=cfg.iters,
    )

    pct = percentiles(times)
    # True wall cost = slowest rank's median.
    p50_max = max_reduce_across_ranks(pct["p50_us"], group=None)
    dtype_bytes = 2 if cfg.dtype_mode == "bf16" else 1
    bpt = dispatch_bytes_per_token(cfg.hidden, cfg.topk, dtype_bytes)
    gbps = achieved_gbps(cfg.num_tokens, bpt, p50_max / 1e6)

    if env.rank == 0:
        result = {
            "backend": cfg.backend,
            "regime": cfg.regime,
            "dtype_mode": cfg.dtype_mode,
            "routing": cfg.routing,
            "world_size": env.world_size,
            "num_tokens": cfg.num_tokens,
            "hidden": cfg.hidden,
            "num_experts": cfg.num_experts,
            "topk": cfg.topk,
            "roundtrip_p50_us": p50_max,
            "roundtrip_p10_us": pct["p10_us"],
            "roundtrip_p90_us": pct["p90_us"],
            "achieved_gbps": gbps,
        }
        with open(cfg.out, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
