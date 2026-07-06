# Internal roadmap / follow-ups

Backend coverage (former README milestones):

- [ ] FlashInfer a2a backend (former Milestone 2)
- [ ] raw-DeepEP cross-check against the sglang-wrapped path (former Milestone 3)
- [ ] Verify Mooncake / NIXL cells on GPU (currently only DeepEP is
      GB300-verified; others rely on the N/A fallback)

Measurement improvements:

- [ ] Balanced-floor field for `rx_skew_max_mean` (~1 + sqrt(2 ln W)/sqrt(tokens*topk));
      report measured skew alongside its multinomial expectation
- [ ] Split `rx_pairs_per_rank` into local vs remote pairs (wire bytes vs
      loopback; matters at small EP)
- [ ] Single-allocation A/B mode in run_all (run all cells inside one
      torchrun/Slurm allocation — allocation variance measured up to ~1.7x)
- [ ] Clock-state control for graph timing (sparse tiny kernels may run at
      idle clocks vs serving's sustained boost)
- [ ] Count quantization-scale bytes when the dispatch output carries scales
      outside the (payload, scales) tuple (<=3% undercount today)

Workload realism:

- [ ] Synthetic expert option: dummy per-rank delay/GEMM proportional to
      rx_pairs, restoring the compute-skew term and combine-send stagger that
      identity experts remove
- [ ] Per-layer trace replay (real captures vary per layer; today one routing
      is replayed for all iterations)
- [ ] Converter from sglang expert_distribution_recorder dumps to the npz
      trace format
