from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNorm


def _fmt_float(x: float) -> str:
    # Match parity_diff_metrics.py style (python float()).
    if x != x:  # nan
        return "nan"
    if x == float("inf"):
        return "inf"
    if x == float("-inf"):
        return "-inf"
    return f"{x:.6g}"


@torch.no_grad()
def _summarize_tensor(name: str, t: torch.Tensor, head_n: int = 16) -> str:
    # Produce one-line summary that parity_diff_metrics.py can parse.
    # Format:
    # <name> shape=(...) dtype=<dtype> mean=... std=... min=... max=... absmax=... head16=[...]
    if t.numel() == 0:
        mean = std = mn = mx = absmax = float("nan")
        head = []
    else:
        tf = t.detach().float()
        mean = tf.mean().item()
        # Use unbiased=False to make it stable across shapes.
        std = tf.std(unbiased=False).item()
        mn = tf.min().item()
        mx = tf.max().item()
        absmax = tf.abs().max().item()
        head = tf.reshape(-1)[:head_n].tolist()
    head_str = ", ".join(_fmt_float(float(v)) for v in head)
    return (
        f"{name} "
        f"shape={tuple(t.shape)} "
        f"dtype={t.dtype} "
        f"mean={_fmt_float(mean)} "
        f"std={_fmt_float(std)} "
        f"min={_fmt_float(mn)} "
        f"max={_fmt_float(mx)} "
        f"absmax={_fmt_float(absmax)} "
        f"head{head_n}=[{head_str}]"
    )


def _make_cases(
    device: str,
    dtype: torch.dtype,
    hidden_size: int,
    batch_sizes: list[int],
    seq_lens: list[int],
    seeds: list[int],
) -> list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]:
    cases: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for seed in seeds:
        # Generate inputs on CPU in FP32 for better cross-env determinism,
        # then move/cast to the requested device/dtype.
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        for b in batch_sizes:
            for s in seq_lens:
                # AdaLN expects:
                # x: (B, S, H)
                # scale, shift: broadcastable to x (common: (B, 1, H) or (B, H))
                x_cpu = torch.randn((b, s, hidden_size), device="cpu", dtype=torch.float32, generator=g)
                scale_cpu = torch.randn((b, 1, hidden_size), device="cpu", dtype=torch.float32, generator=g)
                shift_cpu = torch.randn((b, 1, hidden_size), device="cpu", dtype=torch.float32, generator=g)

                x = x_cpu.to(device=device, dtype=dtype)
                scale = scale_cpu.to(device=device, dtype=dtype)
                shift = shift_cpu.to(device=device, dtype=dtype)
                tag = f"seed{seed}.b{b}.s{s}.h{hidden_size}"
                cases.append((tag, x, scale, shift))
    return cases


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Dump deterministic AdaLayerNorm I/O summaries for cross-env parity diff. "
            "Run in two environments and compare outputs using parity_diff_metrics.py."
        )
    )
    ap.add_argument("--out", required=True, type=Path, help="Output .log path.")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to run on.")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="Input dtype.")
    ap.add_argument("--hidden-size", type=int, default=3584, help="Hidden size H.")
    ap.add_argument("--batch-sizes", default="1,2,4", help="Comma-separated batch sizes.")
    ap.add_argument("--seq-lens", default="1,8,64,256", help="Comma-separated seq lens.")
    ap.add_argument("--seeds", default="0,1,2", help="Comma-separated RNG seeds.")
    ap.add_argument("--head-n", type=int, default=16, help="How many head elements to log.")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but --device=cuda was requested.")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = args.device

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    # Module under test
    adaln = AdaLayerNorm(args.hidden_size, elementwise_affine=False, eps=1e-6).to(device=device, dtype=dtype).eval()

    cases = _make_cases(
        device=device,
        dtype=dtype,
        hidden_size=args.hidden_size,
        batch_sizes=batch_sizes,
        seq_lens=seq_lens,
        seeds=seeds,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write(f"# adalayernorm_parity_dump\n")
        f.write(f"# device={device}\n")
        f.write(f"# dtype={dtype}\n")
        f.write(f"# hidden_size={args.hidden_size}\n")
        f.write(
            "# VLLM_OMNI_DIFFUSION_FORCE_FP32_NORMS="
            + os.environ.get("VLLM_OMNI_DIFFUSION_FORCE_FP32_NORMS", "")
            + "\n"
        )
        for i, (tag, x, scale, shift) in enumerate(cases):
            prefix = f"adaln.case{i}.{tag}"
            y = adaln.forward_native(x, scale, shift)
            f.write(_summarize_tensor(f"{prefix}.x", x, head_n=args.head_n) + "\n")
            f.write(_summarize_tensor(f"{prefix}.scale", scale, head_n=args.head_n) + "\n")
            f.write(_summarize_tensor(f"{prefix}.shift", shift, head_n=args.head_n) + "\n")
            f.write(_summarize_tensor(f"{prefix}.y", y, head_n=args.head_n) + "\n")


if __name__ == "__main__":
    main()

