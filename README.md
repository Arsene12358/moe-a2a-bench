# moe-a2a-bench

Cross-backend **all2all** (MoE dispatch/combine) performance suite for SGLang
large-scale expert parallelism. It drives the *production*
`create_moe_dispatcher` path (`BaseDispatcher.dispatch` / `combine`), so
numbers reflect what serving actually pays. Expert compute is identity, so
only the a2a cycle is measured — dispatch + combine, with optional per-phase
and transport-only timing.

This is a **standalone repo**: it imports `sglang` as an installed dependency,
so SGLang (plus the backend libraries DeepEP / Mooncake / NIXL) must be
available in the runtime environment / container.

## Repository layout

```
ep_a2a/
  config.py      # BenchConfig + CLI (workload, routing, timing knobs)
  workload.py    # routing generators (balanced/hotspot/zipf) + real-trace replay
  bootstrap.py   # torch.distributed + SGLang parallel state + dispatcher build
  adapter.py     # dispatch -> identity -> combine glue, wire-format probe,
                 # correctness gate
  metrics.py     # critical path, per-rank spread, rx-load counting, bandwidth
  timing.py      # CUDA-event timing: eager and CUDA-graph-replay modes
  bench.py       # one benchmark cell -> result JSON
  report.py      # aggregate result JSONs -> comparison table
  run_all.py     # sweep orchestrator over (backend, regime, dtype, routing)
tests/
  test_units.py  # CPU-only unit tests for the pure helpers
TODO.md          # internal roadmap / follow-ups
```

## What a cell measures

One benchmark cell = (backend, regime, dtype-mode, routing) on a fixed EP
group (`ep_size == world_size`, experts placed contiguously,
`num_experts / world_size` per rank):

1. Build the production dispatcher via sglang's factory.
2. Generate one routing workload (see routing modes) and probe the actual
   wire format from a real dispatch output.
3. Time the dispatch/combine cycle (eager or graph-replay; whole roundtrip or
   per phase).
4. Gather the rank×iter timing matrix and routed-load counts; emit one JSON.

## Running

### Install

In a GPU environment that already has `sglang` + `torch` + the backend libs:

```bash
pip install -e .[dev]
```

### Single cell (torchrun)

```bash
torchrun --nproc_per_node=4 -m ep_a2a.bench \
  --backend deepep --regime decode --dtype-mode bf16 \
  --routing balanced --num-tokens 512 \
  --split-phases --cuda-graph --out result.json
```

Expected stdout: `correctness gate: OK` followed by the result JSON.
Transport-only numbers come from `--split-phases --cuda-graph` (see timing
modes); drop both for the eager production-path roundtrip.

### Full sweep

```bash
# single node, 8 GPUs
GPUS_PER_NODE=8 python -m ep_a2a.run_all

# choose axes explicitly
python -m ep_a2a.run_all --backends deepep --regimes decode \
  --routings balanced zipf --skew 1.2 --cuda-graph
```

`run_all` launches one torchrun per cell and prints a comparison table from
the result JSONs (also available standalone: `python -m ep_a2a.report`).

### Multi-node

Topology comes from env; run the same command on every node:

```bash
MASTER_ADDR=<node0> MASTER_PORT=8899 NNODES=4 NODE_RANK=$THIS_NODE \
GPUS_PER_NODE=4 python -m ep_a2a.run_all
```

Slurm sketch (one task per node inside a container image that has the stack):

```bash
sbatch -N 4 --gpus-per-node=4 --wrap "srun --ntasks=4 --ntasks-per-node=1 \
  --container-image=<sglang.sqsh> --container-mounts=<repo>:/w \
  bash -c 'cd /w && MASTER_ADDR=\$SLURM_LAUNCH_NODE_IPADDR MASTER_PORT=8899 \
    torchrun --nnodes=\$SLURM_JOB_NUM_NODES --node_rank=\$SLURM_NODEID \
    --nproc_per_node=4 --master_addr=\$MASTER_ADDR --master_port=\$MASTER_PORT \
    -m ep_a2a.bench --backend deepep --regime decode --dtype-mode bf16 \
    --split-phases --cuda-graph --out /w/result.json'"
```

### Cluster environment knobs

- `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` — DeepEP low-latency
  buffer geometry. Size it to your workload: it is a real lever on the LL
  kernel floor (~25% observed), and it is recorded in the result JSON.
- `NVSHMEM_REMOTE_TRANSPORT=none` — required on clusters whose RDMA NICs are
  not mlx5 when the EP group stays inside one NVLink domain (e.g. GB200/GB300
  NVL72 racks); NVSHMEM otherwise crashes probing IB.
- `MASTER_PORT` — pick a port outside your cluster's ephemeral range
  (`sysctl net.ipv4.ip_local_port_range`); some HPC clusters widen it and
  random source ports can collide with listeners.
- `EP_A2A_USE_FABRIC=1` / `EP_A2A_DISABLE_MNNVL=1` — DeepEP Buffer overrides
  for cross-node bring-up on fabrics that need explicit fabric handles; only
  injected if this DeepEP build accepts the kwargs.

### Unit tests (CPU)

```bash
pytest tests/ -v
```

## Routing modes (expert load balance)

a2a cost is a function of expert load skew, so routing is a first-class axis
(`--routing`, sweepable via `run_all --routings ...`):

- `balanced` — uniform k-subsets (default).
- `hotspot` — legacy worst case: the first `num_experts//8` experts win every
  token (`imbalanced` is an accepted alias). A corner, not a dial.
- `zipf` — calibrated skew: expert popularity ∝ 1/(rank+1)^`--skew`, sampled
  per token without replacement (Gumbel-top-k). `--hot-placement scattered`
  (default) places experts load-aware greedily across ranks (best-case static
  placement, an EPLB-like envelope); `contiguous` piles the popular experts
  onto the first ranks (worst case). Real unoptimized placements fall between.
- `trace` — replay real routing from `--trace-path file.npz` containing
  `topk_ids [N, topk] int64` (+ optional `topk_weights float32`), e.g.
  converted from sglang's `expert_distribution_recorder`. Rank r takes rows
  `[r*num_tokens, ...)` with wraparound. This is the ground-truth mode: it
  reproduces production skew exactly.

Every result records the *achieved* load, not just the intended one:
`rx_pairs_per_rank` (routed (token, expert) pairs received per rank — what
sizes both the receive traffic and the expert GEMM), `rx_unique_tokens_per_rank`
(for backends that dedup per destination), and `rx_skew_max_mean`.

## Timing modes

The `timing` field records which mode produced a result:

- **eager** (default): times the production eager code path — kernels plus
  the wrapper's host work (python, launches, count read-backs). Real cost for
  non-graphed serving, but not transport.
- **`--cuda-graph`** (decode regime only): captures each timed region into a
  CUDA graph and times replays — the device path only, comparable to
  graph-mode serving and to kernel sums in nsys traces. Prefill's normal mode
  host-syncs internally and cannot be captured.

`--split-phases` times the pure dispatch and combine kernels (whose fused
receive-waits are genuine transport). The identity materialization — the
stand-in for the expert GEMM's output write, a cast over the whole padded
dispatch buffer — runs once at setup and is excluded: it is compute-side work
that dwarfs the a2a kernels, and production pays it inside the masked GEMM,
not the transport. The *unsplit* roundtrip still includes it every iteration.
**Use `--split-phases --cuda-graph` for transport-only numbers.** The
send/recv halves *within* a phase are fused in the kernels and are not
separable from the host; attribute them by correlating
`{dispatch,combine}_per_rank_p50_us` against `rx_pairs_per_rank` instead.

## Result fields

Latency (µs): `critical_path_p10/50/90_us` — per-iteration max across ranks,
then percentiles (`roundtrip_p50_us` = the p50, kept for the table);
`dispatch_p50_us` / `combine_p50_us` (+ per-rank lists) with
`--split-phases`.

Attribution: `per_rank_p50_us`, `rank_spread` (max/min of per-rank medians),
`straggler_stability` (fraction of iterations whose slowest rank is the modal
slowest — ~1.0 = statically hot rank, ~1/world_size = rotating jitter).

Load: `rx_pairs_per_rank`, `rx_unique_tokens_per_rank`, `rx_skew_max_mean`.

Bandwidth: `achieved_gbps` (send-side, from actual wire bytes),
`hot_rank_rx_gbps` (hottest rank's receive — the lane that saturates first
under skew).

Wire truth & provenance: the dtype MODE is advisory across sglang versions,
so the JSON records what was probed from a real dispatch output —
`dispatch_wire_dtype`, `combine_wire_dtype`, `dispatch_wire_bytes_per_token`
(payload + amortized quant scales when carried in the output tuple) — plus
`nodelist` and `deepep_max_dispatch_tokens_per_rank`. Results are not
comparable without the provenance: allocation-to-allocation variance up to
~1.7x and buffer geometry ~25% have been measured on the same config.

## Notes

- `num_experts` must be divisible by world size (ep_size).
- A backend whose library is not installed is reported as `N/A`.
- `--dtype-mode native` requires fp8 dispatch to be active (JIT DeepGEMM); if
  it cannot be forced, that cell should be skipped rather than reported as
  bf16.
- The workload (including routing) is generated once and replayed for all
  iterations; there is no step-to-step routing variance like real serving.
- Compare configs within a single allocation where possible (one job running
  cells back-to-back), and prefer medians over means — LL kernel times are
  bimodal (fast phase + receive-wait phase).
