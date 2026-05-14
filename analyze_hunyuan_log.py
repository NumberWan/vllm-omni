#!/usr/bin/env python
"""
Analyze HunyuanImage-3.0-Instruct multi-stage diffusion metrics from API logs.

Usage:
  # 1) 跑 benchmark，例如：
  #   python benchmarks/diffusion/diffusion_benchmark_serving.py \
  #     --backend openai \
  #     --base-url http://127.0.0.1:8094 \
  #     --model "$SNAP" \
  #     --task t2i \
  #     --dataset random \
  #     --num-prompts 10 \
  #     --width 1024 --height 1024 \
  #     --num-inference-steps 8
  #
  # 2) 把 server log 複製到一個檔，例如：
  #   cp /root/.cursor/projects/public-wtk/terminals/1.txt hunyuan_server.log
  #
  # 3) 生成 CSV（可用 Excel 打開）：
  #   python analyze_hunyuan_log.py hunyuan_server.log hunyuan_metrics.csv
"""

from __future__ import annotations

import csv
import json
import re
import sys
from statistics import mean
from typing import Any


LINE_RE = re.compile(r"\[hunyuan_image3_metrics\]\s+({.+})")


def _parse_log(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            try:
                data = json.loads(m.group(1))
            except Exception:
                continue

            sd = data.get("stage_durations_ms") or {}
            # Orchestrator uses stage_0_gen_ms / stage_1_gen_ms (milliseconds).
            ar_ms = (
                sd.get("stage_0_gen_ms")
                or sd.get("stage_0")
                or sd.get("ar")
                or sd.get("AR")
                or 0.0
            )
            dit_ms = (
                sd.get("stage_1_gen_ms")
                or sd.get("stage_1")
                or sd.get("dit")
                or sd.get("DiT")
                or 0.0
            )
            total_ms = float(ar_ms) + float(dit_ms)

            rows.append(
                {
                    "request_id": data.get("request_id"),
                    "model": data.get("model"),
                    "width": data.get("width"),
                    "height": data.get("height"),
                    "size": data.get("size"),
                    "num_inference_steps": data.get("num_inference_steps"),
                    "guidance_scale": data.get("guidance_scale"),
                    "true_cfg_scale": data.get("true_cfg_scale"),
                    "n": data.get("n"),
                    "seed": data.get("seed"),
                    "prompt_words": data.get("prompt_words"),
                    "ar_ms": ar_ms,
                    "dit_ms": dit_ms,
                    "total_ms": total_ms or None,
                }
            )
    return rows


def _write_csv(rows: list[dict[str, Any]], out_path: str) -> None:
    if not rows:
        print("No hunyuan_image3_metrics entries found.", file=sys.stderr)
        return

    fieldnames = [
        "request_id",
        "model",
        "width",
        "height",
        "size",
        "num_inference_steps",
        "guidance_scale",
        "true_cfg_scale",
        "n",
        "seed",
        "prompt_words",
        "ar_ms",
        "dit_ms",
        "total_ms",
    ]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # 簡單在 stdout 印 summary，方便你對比不同 run。
    ar_list = [float(r["ar_ms"]) for r in rows if r["ar_ms"]]
    dit_list = [float(r["dit_ms"]) for r in rows if r["dit_ms"]]
    tot_list = [float(r["total_ms"]) for r in rows if r["total_ms"]]
    if ar_list and dit_list and tot_list:
        print(f"Samples: {len(rows)}")
        print(f"AR mean latency:  {mean(ar_list):.1f} ms")
        print(f"DiT mean latency: {mean(dit_list):.1f} ms")
        print(f"Total mean:       {mean(tot_list):.1f} ms")
    print(f"Wrote CSV to {out_path}")


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        print(f"Usage: {argv[0]} LOG_PATH OUT_CSV", file=sys.stderr)
        sys.exit(1)
    log_path, out_csv = argv[1], argv[2]
    rows = _parse_log(log_path)
    _write_csv(rows, out_csv)


if __name__ == "__main__":
    main(sys.argv)

