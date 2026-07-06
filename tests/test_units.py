"""CPU-only unit tests for the EP all2all benchmark helpers.

The GPU/distributed pieces (bootstrap/adapter/timing/bench) are validated by the
2-GPU smoke run documented in the README, not here.
"""
import pytest
import torch

from ep_a2a.config import BenchConfig, parse_args
from ep_a2a.metrics import achieved_gbps, dispatch_bytes_per_token
from ep_a2a.report import format_table


def test_benchconfig_defaults():
    cfg = BenchConfig(backend="deepep", regime="prefill", dtype_mode="bf16")
    assert cfg.hidden == 7168
    assert cfg.num_experts == 256
    assert cfg.topk == 8
    assert cfg.num_tokens == 4096  # prefill default
    assert cfg.warmups == 20 and cfg.iters == 30


def test_benchconfig_decode_token_default():
    cfg = BenchConfig(backend="deepep", regime="decode", dtype_mode="bf16")
    assert cfg.num_tokens == 128  # decode default


def test_parse_args_minimal():
    cfg = parse_args(
        ["--backend", "nixl", "--regime", "decode", "--dtype-mode", "native"]
    )
    assert cfg.backend == "nixl"
    assert cfg.regime == "decode"
    assert cfg.dtype_mode == "native"


def test_dispatch_bytes_per_token_bf16():
    # hidden=7168, topk=8, bf16 (2 bytes): each token is sent to topk ranks.
    assert dispatch_bytes_per_token(hidden=7168, topk=8, dtype_bytes=2) == 7168 * 8 * 2


def test_dispatch_wire_bytes_per_token():
    from ep_a2a.metrics import dispatch_wire_bytes_per_token

    # fp8 payload (1B) + fp32 scale per 128 elements (0.03125 B/elem)
    got = dispatch_wire_bytes_per_token(6144, 8, 1, 4 / 128)
    assert got == 6144 * 8 * (1 + 0.03125)
    # bf16, no scales
    assert dispatch_wire_bytes_per_token(7168, 8, 2, 0.0) == 7168 * 8 * 2


def test_achieved_gbps():
    # 1000 tokens, 1024 bytes/token, 1 ms.
    gbps = achieved_gbps(num_tokens=1000, bytes_per_token=1024, seconds=1e-3)
    assert abs(gbps - (1000 * 1024) / 1e-3 / 1e9) < 1e-9


def test_format_table_orders_and_marks_na():
    results = [
        {
            "backend": "deepep",
            "regime": "prefill",
            "dtype_mode": "bf16",
            "roundtrip_p50_us": 120.0,
            "achieved_gbps": 350.0,
        },
        {
            "backend": "nixl",
            "regime": "prefill",
            "dtype_mode": "bf16",
            "status": "unavailable",
            "reason": "no nixl",
        },
    ]
    table = format_table(results)
    assert "deepep" in table and "nixl" in table
    assert "120.0" in table
    assert "N/A" in table  # unavailable backend shown as N/A


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("routing", ["balanced", "imbalanced", "zipf"])
def test_make_workload_shapes(routing):
    from ep_a2a.workload import make_workload

    cfg = BenchConfig(
        backend="deepep",
        regime="prefill",
        dtype_mode="bf16",
        num_tokens=64,
        hidden=256,
        num_experts=16,
        topk=4,
        routing=routing,
    )
    hidden_states, topk_output = make_workload(cfg, rank=0, world_size=4, device="cuda")
    assert hidden_states.shape == (64, 256)
    assert hidden_states.dtype == torch.bfloat16
    assert topk_output.topk_ids.shape == (64, 4)
    assert topk_output.topk_weights.shape == (64, 4)
    assert int(topk_output.topk_ids.min()) >= 0
    assert int(topk_output.topk_ids.max()) < 16
    row = topk_output.topk_ids[0].tolist()
    assert len(set(row)) == len(row)


# ---------------------------------------------------------------------------
# config: new routing knobs
# ---------------------------------------------------------------------------


def test_config_cuda_graph_decode_only():
    cfg = BenchConfig(
        backend="deepep", regime="decode", dtype_mode="bf16", cuda_graph=True
    )
    assert cfg.cuda_graph
    with pytest.raises(ValueError, match="cuda_graph"):
        BenchConfig(
            backend="deepep", regime="prefill", dtype_mode="bf16", cuda_graph=True
        )


def test_config_imbalanced_alias_and_trace_validation():
    cfg = BenchConfig(
        backend="deepep", regime="decode", dtype_mode="bf16", routing="imbalanced"
    )
    assert cfg.routing == "hotspot"
    with pytest.raises(ValueError, match="trace_path"):
        BenchConfig(
            backend="deepep", regime="decode", dtype_mode="bf16", routing="trace"
        )


def test_parse_args_routing_knobs(tmp_path):
    trace = tmp_path / "t.npz"
    trace.write_bytes(b"")  # existence not checked at parse time
    cfg = parse_args(
        [
            "--backend", "deepep", "--regime", "decode", "--dtype-mode", "bf16",
            "--routing", "zipf", "--skew", "1.4", "--hot-placement", "contiguous",
            "--split-phases",
        ]
    )
    assert cfg.routing == "zipf" and cfg.skew == 1.4
    assert cfg.hot_placement == "contiguous"
    assert cfg.split_phases
    cfg2 = parse_args(
        [
            "--backend", "deepep", "--regime", "decode", "--dtype-mode", "bf16",
            "--routing", "trace", "--trace-path", str(trace),
        ]
    )
    assert cfg2.routing == "trace" and cfg2.trace_path == str(trace)


# ---------------------------------------------------------------------------
# metrics: rank x iter matrix analysis
# ---------------------------------------------------------------------------


def test_critical_path_vs_max_of_p50s():
    from ep_a2a.metrics import critical_path_series, per_rank_p50s

    # Alternating stragglers: every iteration's wall time is 20, but each
    # rank's own median is 15 -> max-of-p50s (the old metric) says 15.
    m = torch.tensor(
        [[10.0, 20.0, 10.0, 20.0], [20.0, 10.0, 20.0, 10.0]], dtype=torch.float64
    )
    crit = critical_path_series(m)
    assert crit.tolist() == [20.0, 20.0, 20.0, 20.0]
    max_of_p50 = max(per_rank_p50s(m))
    assert max_of_p50 == 15.0  # the underestimate #2 exists to fix


def test_rank_spread_and_straggler_stability():
    from ep_a2a.metrics import rank_spread, straggler_stability

    assert rank_spread([10.0, 10.0, 10.0]) == 1.0
    assert rank_spread([10.0, 25.0]) == 2.5

    # statically hot rank 1 -> stability 1.0
    static = torch.tensor([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]], dtype=torch.float64)
    assert straggler_stability(static) == 1.0
    # rotating straggler -> stability well below 1
    rotating = torch.tensor(
        [[3.0, 1.0, 1.0, 3.0], [1.0, 3.0, 3.0, 1.0]], dtype=torch.float64
    )
    assert straggler_stability(rotating) == 0.5


def test_rx_counting_hand_example():
    from ep_a2a.metrics import (
        rx_pairs_per_rank,
        rx_unique_tokens_per_rank,
        skew_max_mean,
    )

    # 8 experts over 4 ranks -> 2 experts/rank; rank r owns {2r, 2r+1}.
    # token0 -> experts 0,1 (both rank 0); token1 -> experts 2,7 (ranks 1,3)
    ids = torch.tensor([[0, 1], [2, 7]])
    pairs = rx_pairs_per_rank(ids, num_experts=8, world_size=4)
    assert pairs.tolist() == [2, 1, 0, 1]
    uniq = rx_unique_tokens_per_rank(ids, num_experts=8, world_size=4)
    assert uniq.tolist() == [1, 1, 0, 1]  # token0 counted once on rank 0
    assert skew_max_mean(pairs) == 2.0  # max 2 / mean 1


# ---------------------------------------------------------------------------
# workload: zipf generator (pure, CPU)
# ---------------------------------------------------------------------------


def test_zipf_probs_uniform_at_zero_skew():
    from ep_a2a.workload import zipf_expert_probs

    p = zipf_expert_probs(16, skew=0.0, world_size=4)
    assert torch.allclose(p, torch.full((16,), 1 / 16.0), atol=1e-6)


def test_zipf_probs_placement():
    from ep_a2a.workload import zipf_expert_probs

    E, W = 16, 4
    cont = zipf_expert_probs(E, skew=1.0, world_size=W, hot_placement="contiguous")
    scat = zipf_expert_probs(E, skew=1.0, world_size=W, hot_placement="scattered")
    assert abs(float(cont.sum()) - 1.0) < 1e-6
    assert abs(float(scat.sum()) - 1.0) < 1e-6
    # contiguous: hottest expert is id 0; per-rank mass is front-loaded
    assert int(cont.argmax()) == 0
    local = E // W
    cont_rank_mass = cont.view(W, local).sum(dim=1)
    cont_ratio = float(cont_rank_mass.max() / cont_rank_mass.min())
    assert cont_ratio > 4.0  # pathological by construction
    # scattered (LPT): far more even, though the zipf head bounds evenness —
    # whichever rank owns the #1 expert carries its full mass.
    scat_rank_mass = scat.view(W, local).sum(dim=1)
    scat_ratio = float(scat_rank_mass.max() / scat_rank_mass.min())
    assert scat_ratio < 2.0
    assert scat_ratio < cont_ratio / 2


def test_gumbel_topk_skew_monotonic():
    from ep_a2a.metrics import rx_pairs_per_rank, skew_max_mean
    from ep_a2a.workload import gumbel_topk_ids, zipf_expert_probs

    E, W, n, k = 64, 8, 4096, 4
    skews = []
    for alpha in (0.0, 0.8, 1.6):
        p = zipf_expert_probs(E, alpha, W, hot_placement="contiguous")
        g = torch.Generator().manual_seed(0)
        scores, ids = gumbel_topk_ids(p, n, k, g)
        assert ids.shape == (n, k)
        assert int(ids.min()) >= 0 and int(ids.max()) < E
        # no duplicate experts within a token
        assert (ids.sort(dim=1).values.diff(dim=1) > 0).all()
        skews.append(skew_max_mean(rx_pairs_per_rank(ids, E, W)))
    # achieved rx skew grows with the zipf dial
    assert skews[0] < skews[1] < skews[2]
    assert skews[0] < 1.15  # balanced ~1.0 with sampling noise


# ---------------------------------------------------------------------------
# workload: trace replay (pure, CPU)
# ---------------------------------------------------------------------------


def _write_trace(path, n=32, topk=4, num_experts=16, with_weights=True):
    import numpy as np

    rng = np.random.default_rng(0)
    ids = np.stack(
        [rng.choice(num_experts, size=topk, replace=False) for _ in range(n)]
    ).astype("int64")
    arrays = {"topk_ids": ids}
    if with_weights:
        arrays["topk_weights"] = rng.random((n, topk)).astype("float32")
    np.savez(path, **arrays)
    return ids


def test_trace_slice_roundtrip_and_wraparound(tmp_path):
    from ep_a2a.workload import load_trace_slice

    path = str(tmp_path / "trace.npz")
    ids = _write_trace(path, n=32, topk=4, num_experts=16)
    got, w = load_trace_slice(path, rank=0, num_tokens=8, topk=4, num_experts=16)
    assert got.tolist() == ids[:8].tolist()
    assert w.shape == (8, 4)
    # rank slicing walks forward; wraparound past N is deterministic
    got3, _ = load_trace_slice(path, rank=3, num_tokens=16, topk=4, num_experts=16)
    assert got3[:16].tolist() == ids[[(48 + i) % 32 for i in range(16)]].tolist()


def test_trace_slice_default_weights(tmp_path):
    from ep_a2a.workload import load_trace_slice

    path = str(tmp_path / "trace.npz")
    _write_trace(path, with_weights=False)
    _, w = load_trace_slice(path, rank=0, num_tokens=4, topk=4, num_experts=16)
    assert torch.allclose(w, torch.full((4, 4), 0.25))


def test_trace_slice_validation(tmp_path):
    import numpy as np

    from ep_a2a.workload import load_trace_slice

    bad_topk = str(tmp_path / "bad_topk.npz")
    _write_trace(bad_topk, topk=2)
    with pytest.raises(ValueError, match="topk"):
        load_trace_slice(bad_topk, rank=0, num_tokens=4, topk=4, num_experts=16)

    bad_range = str(tmp_path / "bad_range.npz")
    np.savez(bad_range, topk_ids=np.array([[0, 99]], dtype="int64"))
    with pytest.raises(ValueError, match="out of"):
        load_trace_slice(bad_range, rank=0, num_tokens=1, topk=2, num_experts=16)

    missing = str(tmp_path / "missing.npz")
    np.savez(missing, other=np.zeros(1))
    with pytest.raises(ValueError, match="topk_ids"):
        load_trace_slice(missing, rank=0, num_tokens=1, topk=2, num_experts=16)
