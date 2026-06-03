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

**Code is UNVERIFIED pending the first GPU smoke** — authored on a host without a
GPU. Pure-Python helpers (config / metrics / report) are unit-tested.

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

## Notes

- `num_experts` must be divisible by world size (ep_size).
- A backend whose library is not installed is reported as `N/A`.
- `--dtype-mode native` requires fp8 dispatch to be active (JIT DeepGEMM); if it
  cannot be forced, that cell should be skipped rather than reported as bf16.
- Timing is the dispatch+combine round trip (per-phase split is a planned
  follow-up).

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
