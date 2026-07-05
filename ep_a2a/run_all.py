"""Launch one torchrun job per (backend, regime, dtype_mode) cell and print the
comparison table. Topology comes from env (MASTER_ADDR/PORT, NNODES, NODE_RANK,
GPUS_PER_NODE) so the same script works intranode and multi-node."""
from __future__ import annotations

import argparse
import os
import subprocess

from ep_a2a.config import BACKENDS, DTYPE_MODES, REGIMES
from ep_a2a.report import format_table, load_results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backends", nargs="+", default=BACKENDS)
    p.add_argument("--regimes", nargs="+", default=REGIMES)
    p.add_argument(
        "--dtype-modes", nargs="+", default=DTYPE_MODES, dest="dtype_modes"
    )
    p.add_argument(
        "--routings", nargs="+", default=["balanced"],
        help="routing modes to sweep (balanced/hotspot/zipf/trace)",
    )
    p.add_argument("--skew", type=float, default=1.0, help="zipf exponent")
    p.add_argument(
        "--hot-placement", default="scattered", dest="hot_placement",
        choices=["contiguous", "scattered"],
    )
    p.add_argument("--trace-path", default=None, dest="trace_path")
    p.add_argument("--split-phases", action="store_true", dest="split_phases")
    p.add_argument("--cuda-graph", action="store_true", dest="cuda_graph")
    p.add_argument(
        "--gpus-per-node",
        type=int,
        default=int(os.environ.get("GPUS_PER_NODE", 8)),
    )
    p.add_argument("--nnodes", type=int, default=int(os.environ.get("NNODES", 1)))
    p.add_argument(
        "--node-rank", type=int, default=int(os.environ.get("NODE_RANK", 0))
    )
    p.add_argument(
        "--master-addr", default=os.environ.get("MASTER_ADDR", "127.0.0.1")
    )
    p.add_argument("--master-port", default=os.environ.get("MASTER_PORT", "29500"))
    p.add_argument("--outdir", default="ep_a2a_results")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for backend in args.backends:
        for regime in args.regimes:
            for dtype_mode in args.dtype_modes:
                for routing in args.routings:
                    tag = f"{backend}_{regime}_{dtype_mode}_{routing}"
                    out = os.path.join(args.outdir, f"result_{tag}.json")
                    cmd = [
                        "torchrun",
                        f"--nnodes={args.nnodes}",
                        f"--node_rank={args.node_rank}",
                        f"--nproc_per_node={args.gpus_per_node}",
                        f"--master_addr={args.master_addr}",
                        f"--master_port={args.master_port}",
                        "-m",
                        "ep_a2a.bench",
                        "--backend",
                        backend,
                        "--regime",
                        regime,
                        "--dtype-mode",
                        dtype_mode,
                        "--routing",
                        routing,
                        "--skew",
                        str(args.skew),
                        "--hot-placement",
                        args.hot_placement,
                        "--out",
                        out,
                    ]
                    if args.trace_path:
                        cmd += ["--trace-path", args.trace_path]
                    if args.split_phases:
                        cmd += ["--split-phases"]
                    if args.cuda_graph and regime == "decode":
                        cmd += ["--cuda-graph"]
                    print(f"[run_all] launching {tag}", flush=True)
                    subprocess.run(cmd, check=False)

    if args.node_rank == 0:
        results = load_results(os.path.join(args.outdir, "result_*.json"))
        print("\n=== EP all2all comparison ===")
        print(format_table(results))


if __name__ == "__main__":
    main()
