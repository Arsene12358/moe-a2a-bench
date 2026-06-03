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
@pytest.mark.parametrize("routing", ["balanced", "imbalanced"])
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
    hidden_states, topk_output = make_workload(cfg, rank=0, device="cuda")
    assert hidden_states.shape == (64, 256)
    assert hidden_states.dtype == torch.bfloat16
    assert topk_output.topk_ids.shape == (64, 4)
    assert topk_output.topk_weights.shape == (64, 4)
    assert int(topk_output.topk_ids.min()) >= 0
    assert int(topk_output.topk_ids.max()) < 16
    row = topk_output.topk_ids[0].tolist()
    assert len(set(row)) == len(row)
