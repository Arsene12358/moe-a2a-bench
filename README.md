# moe-a2a-bench

Cross-backend **all2all** (MoE dispatch/combine) performance suite for SGLang
large-scale expert parallelism (ep_size > 8). Compares backends to help pick one.

It drives the *production* `create_moe_dispatcher` path (`BaseDispatcher.dispatch`
/ `combine`), so numbers reflect what serving actually pays. Expert compute is
identity, so only transport is timed — the dispatch + combine **round trip**.

This is a **standalone repo**: it imports `sglang` as an installed dependency, so
SGLang (plus the backend libraries DeepEP / Mooncake / NIXL) must be available in
the runtime environment / container.

## Status

Milestone 1: **DeepEP, Mooncake, NIXL**. FlashInfer = Milestone 2; raw-DeepEP
cross-check = Milestone 3.

**DeepEP path verified on GB300** (1 node x 4 GPUs, sglang v0.5.11 container):
decode + prefill regimes, all routing modes, and --split-phases (dispatch +
combine sum within 1% of the unsplit roundtrip). Mooncake / NIXL cells and
multi-node remain unverified. Pure-Python helpers are unit-tested.

## Install

In a GPU environment that already has `sglang` + `torch` + the backend libs:

```bash
pip install -e .[dev]
```

## Intranode smoke (2 GPUs)

```bash
torchrun --nproc_per_node=2 -m ep_a2a.bench \
  --backend deepep --regime prefill --dtype-mode bf16 \
  --num-tokens 512 --hidden 2048 --num-experts 16 --topk 4 --out /tmp/r.json
```

Expected: stdout contains `correctness gate: OK` and a JSON result with positive
`roundtrip_p50_us` and `achieved_gbps`.

## Full sweep (single node, 8 GPUs)

```bash
GPUS_PER_NODE=8 python -m ep_a2a.run_all
```

## Multi-node (run on every node; ep_size = nnodes * gpus_per_node)

```bash
MASTER_ADDR=<node0> MASTER_PORT=29500 NNODES=4 NODE_RANK=$THIS_NODE \
GPUS_PER_NODE=8 python -m ep_a2a.run_all
```

## Unit tests (CPU)

```bash
pytest tests/ -v
```

## Expert load balance

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

Timing is reported as the per-iteration **critical path**: the max across
ranks is taken within each iteration, then percentiles over iterations
(`critical_path_p10/50/90_us`; `roundtrip_p50_us` equals the p50 for table
compatibility). Per-rank medians (`per_rank_p50_us`), their `rank_spread`
(max/min), and `straggler_stability` (fraction of iterations whose slowest
rank is the modal slowest — ~1.0 means a statically hot rank, ~1/world_size
means rotating jitter) attribute the cost. `hot_rank_rx_gbps` is the hottest
rank's receive bandwidth — the lane that saturates first under skew.

`--split-phases` times the pure dispatch and combine kernels (whose fused
receive-waits are genuine transport). The identity materialization — the
stand-in for the expert GEMM's output write, a cast over the whole padded
dispatch buffer — runs once at setup and is excluded: it is compute-side
work that dwarfs the a2a kernels (~2.4 ms vs ~0.1 ms at 512 tokens on GB300)
and production pays it inside the masked GEMM, not the transport. The
*unsplit* roundtrip still includes it every iteration (it times the full
dispatcher->identity-expert->combine cycle). Use `--split-phases
--cuda-graph` for transport-only numbers. Note the send/recv halves *within*
a phase are fused in the kernels and are not separable from the host;
attribute them by correlating `{dispatch,combine}_per_rank_p50_us` against
`rx_pairs_per_rank` instead.

Two timing modes (the `timing` field records which one produced a result):

- **eager** (default): times the production eager code path — kernels plus
  the wrapper's host work (python, launches, count read-backs). On the LL
  path the host side dominates (~20x the kernels), which is real cost for
  non-graphed serving but not transport.
- **`--cuda-graph`** (decode regime only): captures each timed region into a
  CUDA graph and times replays — the device path only, comparable to
  graph-mode serving and to kernel sums in nsys traces. Prefill's normal
  mode host-syncs internally and cannot be captured.

## Notes

- `num_experts` must be divisible by world size (ep_size).
- A backend whose library is not installed is reported as `N/A`.
- `--dtype-mode native` requires fp8 dispatch to be active (JIT DeepGEMM); if it
  cannot be forced, that cell should be skipped rather than reported as bf16.
- The workload (including routing) is generated once and replayed for all
  iterations; there is no step-to-step routing variance like real serving.

## Layout

```
ep_a2a/
  config.py      # BenchConfig + CLI
  workload.py    # synthetic hidden_states + StandardTopKOutput
  bootstrap.py   # torch.distributed + SGLang parallel state + dispatcher build
  adapter.py     # dispatch -> identity -> combine glue + correctness gate
  metrics.py     # bandwidth / percentile / cross-rank reduce
  timing.py      # CUDA-event timing
  bench.py       # one (backend, regime, dtype) job -> result JSON
  report.py      # aggregate result JSONs -> comparison table
  run_all.py     # orchestrator over cells
tests/
  test_units.py  # CPU-only unit tests
```
