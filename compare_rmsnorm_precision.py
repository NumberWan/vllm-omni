#!/usr/bin/env python3
"""
Compare RMSNorm output: BF16 fused kernel vs FP32 torch native.

Loads the real Qwen-Image txt_norm.weight and runs both paths
over the same fixed random input (1, 23, 3072).

Usage:
    source /rebase/venv_0.20.0/bin/activate
    python /rebase/compare_rmsnorm_precision.py
"""

import json
import os
import sys

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors

# ── Config ──────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen-Image-2512"
WEIGHT_NAME = "txt_norm.weight"
SEQ_LEN = 23
BATCH = 1
EPS = 1e-6
SEED = 42


# ── Torch native RMSNorm (matches vLLM RMSNorm.forward_static) ──
def rms_norm_torch_native(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Pure PyTorch FP32 RMSNorm — equivalent to vLLM RMSNorm.forward_static."""
    orig_dtype = x.dtype
    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    out = x_fp32 * torch.rsqrt(variance + eps)
    out = out * weight.float()
    return out.to(orig_dtype)


# ── vLLM fused CUDA kernel ──
def rms_norm_fused(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """vLLM fused CUDA RMSNorm — uses vllm._custom_ops.rms_norm."""
    from vllm._custom_ops import rms_norm as fused_rms_norm

    orig_shape = x.shape
    hidden_size = orig_shape[-1]
    x_2d = x.reshape(-1, hidden_size)
    out = torch.empty_like(x_2d)
    fused_rms_norm(out, x_2d, weight, eps)
    return out.reshape(orig_shape)


def load_txt_norm_weight(model_id: str, weight_name: str) -> tuple[torch.Tensor, str]:
    """Download index + shard from HF Hub, return weight and cache path."""
    from huggingface_hub import hf_hub_download

    # Download index to find which shard contains the weight
    print(f"Downloading index for {model_id}...")
    index_path = hf_hub_download(
        model_id,
        "transformer/diffusion_pytorch_model.safetensors.index.json",
    )
    with open(index_path) as f:
        index = json.load(f)
    shard_name = index["weight_map"][weight_name]
    print(f"  {weight_name} -> {shard_name}")

    # Download the specific shard
    print(f"Downloading {shard_name}...")
    shard_path = hf_hub_download(model_id, f"transformer/{shard_name}")
    tensors = load_safetensors(shard_path)
    return tensors[weight_name], shard_path


def compare(x: torch.Tensor, weight: torch.Tensor, eps: float, hidden_size: int, atol: float = 1e-5):
    """Run fused (BF16) vs native (FP32) and print comparison."""
    # Ensure weight on same device as x
    weight = weight.to(device=x.device)

    # 1) Fused CUDA path (input stays in BF16, kernel runs in BF16/FP16 internally)
    with torch.inference_mode():
        out_fused = rms_norm_fused(x.clone(), weight, eps)

    # 2) Torch native FP32 path
    with torch.inference_mode():
        out_native = rms_norm_torch_native(x.clone(), weight, eps)

    # 3) vLLM RMSNorm class via IR ops (default priority — whatever the env gives)
    # Must set up a minimal VllmConfig context for CustomOp dispatch
    from vllm.config import VllmConfig
    from vllm.config.kernel import IrOpPriorityConfig
    from vllm.config.vllm import set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm

    vllm_config = VllmConfig()
    vllm_config.model_config = None  # not needed for RMSNorm op dispatch
    with set_current_vllm_config(vllm_config):
        norm_default = RMSNorm(hidden_size, eps=eps, force_fp32=False)
    norm_default.weight.data.copy_(weight)
    norm_default = norm_default.to(device=x.device, dtype=x.dtype)
    with torch.inference_mode():
        out_vllm_default = norm_default.forward(x.clone())

    # 4) vLLM RMSNorm with rms_norm=['vllm_c'] — the main-branch CI path
    vllm_config_c = VllmConfig()
    vllm_config_c.model_config = None
    vllm_config_c.kernel_config.ir_op_priority = IrOpPriorityConfig.with_default(
        ["vllm_c"], rms_norm=["vllm_c"]
    )
    with set_current_vllm_config(vllm_config_c):
        norm_vllm_c = RMSNorm(hidden_size, eps=eps, force_fp32=False)
    norm_vllm_c.weight.data.copy_(weight)
    norm_vllm_c = norm_vllm_c.to(device=x.device, dtype=x.dtype)
    with torch.inference_mode():
        out_vllm_c = norm_vllm_c.forward(x.clone())

    # ── Print results ──
    print("=" * 70)
    print(f"Input:          {list(x.shape)}  dtype={x.dtype}  device={x.device}")
    print(f"Weight:         {list(weight.shape)}  dtype={weight.dtype}  "
          f"min={weight.min():.6f}  max={weight.max():.6f}")
    print(f"Epsilon:        {eps}")

    # Stats
    diff_fused_native = (out_fused.float() - out_native.float()).abs()
    print(f"\n{'Path':<32} {'Mean':<16} {'Std':<16} {'Min':<16} {'Max':<16}")
    print("-" * 80)
    for label, out in [
        ("fused (_custom_ops.rms_norm)", out_fused),
        ("torch native FP32", out_native),
        ("vLLM RMSNorm (default ir op)", out_vllm_default),
        ("vLLM RMSNorm (vllm_c only)", out_vllm_c),
    ]:
        print(f"{label:<32} {out.float().mean():.10f}  {out.float().std():.10f}  "
              f"{out.float().min():.10f}  {out.float().max():.10f}")

    # Pairwise diffs
    pairs = [
        ("fused vs native", out_fused, out_native),
        ("fused vs vllm_c", out_fused, out_vllm_c),
        ("native vs vllm_c", out_native, out_vllm_c),
        ("fused vs vLLM default", out_fused, out_vllm_default),
        ("vllm_c vs vLLM default", out_vllm_c, out_vllm_default),
    ]
    print(f"\n--- Pairwise diffs (|a - b|) ---")
    for name, a, b in pairs:
        d = (a.float() - b.float()).abs()
        print(f"  {name:<28} max={d.max():.12f}  mean={d.mean():.12f}")

    # Check if fused-vs-native diff exceeds tolerance
    failed = (diff_fused_native > atol).any()
    print(f"\nPass (fused vs native max|diff| < {atol}): {not failed}")
    if failed:
        num_exceed = (diff_fused_native > atol).sum().item()
        print(f"  Elements exceeding tolerance: {num_exceed}/{diff_fused_native.numel()} "
              f"({100*num_exceed/diff_fused_native.numel():.4f}%)")
        top_vals, top_idx = diff_fused_native.flatten().topk(min(5, num_exceed))
        print("  Top-5 worst mismatches:")
        for i in range(len(top_vals)):
            idx = torch.unravel_index(top_idx[i], diff_fused_native.shape)
            print(f"    pos={list(idx)} fused={out_fused[idx].item():.10f} "
                  f"native={out_native[idx].item():.10f} diff={top_vals[i].item():.12f}")

    return out_fused, out_native, out_vllm_default, out_vllm_c


def main():
    torch.manual_seed(SEED)

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # 1) Load real weight from checkpoint (downloads from HF Hub if needed)
    weight, cache_path = load_txt_norm_weight(MODEL_ID, WEIGHT_NAME)
    hidden_size = weight.shape[0]  # auto-detect from actual weight
    print(f"  Weight: shape={list(weight.shape)}  dtype={weight.dtype}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Cache: {cache_path}\n")

    # 2) Create fixed random input in BF16
    x = torch.randn(BATCH, SEQ_LEN, hidden_size, dtype=torch.bfloat16, device=device)

    # 3) Compare and get outputs
    out_fused, out_native, out_vllm_default, out_vllm_c = compare(x, weight, EPS, hidden_size=hidden_size)

    # 4) Save to .pt
    out_path = os.path.join(os.path.dirname(__file__), "txt_norm_comparison.pt")
    torch.save({
        "input": x.cpu(),
        "weight": weight.cpu(),
        "out_fused": out_fused.cpu(),
        "out_native": out_native.cpu(),
        "out_vllm_default": out_vllm_default.cpu(),
        "out_vllm_c": out_vllm_c.cpu(),
        "config": {
            "model": MODEL_ID,
            "weight_name": WEIGHT_NAME,
            "hidden_size": hidden_size,
            "eps": EPS,
            "seed": SEED,
            "shape": list(x.shape),
            "dtype": str(x.dtype),
        },
    }, out_path)
    print(f"\nSaved all outputs to: {out_path}")


if __name__ == "__main__":
    main()
