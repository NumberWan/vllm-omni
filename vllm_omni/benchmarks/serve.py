import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

from vllm.benchmarks.serve import main_async

# Import patch to register daily-omni dataset and omni backends
# This monkey-patches vllm.benchmarks.datasets.get_samples before it's used
# Must be imported before any vllm.benchmarks module usage
import vllm_omni.benchmarks.patch.patch  # noqa: F401
from vllm_omni.benchmarks.patch.patch import (
    maybe_enable_stage_metrics,
    set_print_stage,
    should_request_stage_metrics,
)


def main(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "seed_tts_wer_eval", False):
        os.environ["SEED_TTS_WER_EVAL"] = "1"
    if getattr(args, "seed_tts_wer_save_items", False):
        os.environ["SEED_TTS_WER_SAVE_ITEMS"] = "1"
    if getattr(args, "daily_omni_save_eval_items", False):
        os.environ["DAILY_OMNI_SAVE_EVAL_ITEMS"] = "1"
    set_print_stage(getattr(args, "print_stage", False))
    args.extra_body = maybe_enable_stage_metrics(
        getattr(args, "extra_body", None),
        enabled=should_request_stage_metrics(args),
    )
    result = asyncio.run(main_async(args))
    if (
        isinstance(result, dict)
        and getattr(args, "save_result", False)
        and getattr(args, "dataset_name", None) == "omniinteract"
        and os.environ.get("OMNIINTERACT_WRITE_BENCH_REPORT", "1").lower() not in ("0", "false", "no")
    ):
        from vllm_omni.benchmarks.format_bench_report import write_bench_reports

        result_dir = Path(getattr(args, "result_dir", ".") or ".")
        json_name = getattr(args, "result_filename", None) or "bench_result.json"
        report_path = write_bench_reports(
            result,
            result_dir=result_dir,
            json_path=result_dir / json_name,
        )
        print(f"Wrote OmniInteract bench report: {report_path}")
    return result
