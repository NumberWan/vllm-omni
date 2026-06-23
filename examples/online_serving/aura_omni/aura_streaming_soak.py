#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""12h AURA streaming soak: boot server, warmup, run full demos, validate, repeat."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.online_serving.aura_omni.aura_streaming_soak_validate import validate_run

PYTHON = Path(os.environ.get("AURA_SOAK_PYTHON", "/public/wtk/.venv/bin/python"))
DEMO = _REPO_ROOT / "examples/online_serving/aura_omni/streaming_video_demo.py"
PORT = 8010
HEALTH_URL = f"http://127.0.0.1:{PORT}/health"

WARMUP_VIDEO = "/public/wtk/AURA/AURA/AURA_bench_eval/StreamingBench/figs/example.mp4"
WARMUP_AUDIO = "/public/wtk/vllm-omni/tests/assets/qwen3_tts/clone_2.wav"
FULL_VIDEO = "/public/wtk/aura_prompts/aura_test.mp4"
FULL_AUDIO_SCHEDULE = [
    "1:/public/wtk/aura_prompts/01_frame_what.wav",
    "20:/public/wtk/aura_prompts/02_pool_notify.wav",
    "38:/public/wtk/aura_prompts/03_scene_changes.wav",
    "48:/public/wtk/aura_prompts/01_frame_what.wav",
]

SERVER_CMD = [
    "vllm",
    "serve",
    "/models/AURA",
    "--omni",
    "--deploy-config",
    "/tmp/aura_omni_gpu23.yaml",
    "--port",
    str(PORT),
    "--trust-remote-code",
    "--served-model-name",
    "aurateam/AURA",
    "--init-timeout",
    "2400",
    "--stage-init-timeout",
    "900",
]


@dataclass
class RunRecord:
    kind: str
    run_id: int
    passed: bool
    reasons: list[str] = field(default_factory=list)
    started_at: str = ""
    ended_at: str = ""
    duration_s: float = 0.0


@dataclass
class SoakState:
    consecutive_pass: int = 0
    total_full_runs: int = 0
    server_boots: int = 0
    runs: list[RunRecord] = field(default_factory=list)
    status: str = "running"
    failure_reason: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str, log_dir: Path) -> None:
    line = f"[{_utc_now()}] {msg}"
    print(line, flush=True)
    with (log_dir / "soak.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _read_server_log(server_log: Path, offset: int) -> str:
    if not server_log.exists():
        return ""
    data = server_log.read_text(encoding="utf-8", errors="replace")
    if offset >= len(data):
        return ""
    return data[offset:]


def _kill_port(port: int) -> None:
    subprocess.run(
        ["fuser", "-k", f"{port}/tcp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    subprocess.run(
        ["pkill", "-f", f"vllm serve /models/AURA.*--port {port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    time.sleep(3)


def _wait_health(timeout_s: float, log_dir: Path) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=5) as resp:
                if resp.status == 200:
                    _log("health check OK", log_dir)
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(10)
    return False


def _run_demo(cmd: list[str], log_path: Path, *, recv_timeout: float) -> tuple[int, str]:
    full_cmd = cmd + ["--recv-timeout", str(recv_timeout)]
    proc = subprocess.run(
        full_cmd,
        cwd=str(_REPO_ROOT),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "2,3")},
        capture_output=True,
        text=True,
        timeout=recv_timeout + 120,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    log_path.write_text(output, encoding="utf-8")
    return proc.returncode, output


def _warmup_cmd() -> list[str]:
    return [
        str(PYTHON),
        str(DEMO),
        "--url",
        f"ws://localhost:{PORT}/v1/video/chat/stream",
        "--model",
        "aurateam/AURA",
        "--video",
        WARMUP_VIDEO,
        "--burst-interval",
        "0",
        "--fps",
        "2",
        "--audio",
        WARMUP_AUDIO,
        "--audio-at-sec",
        "1",
        "--max-duration",
        "8",
        "--no-evs",
    ]


def _full_cmd() -> list[str]:
    cmd = [
        str(PYTHON),
        str(DEMO),
        "--url",
        f"ws://localhost:{PORT}/v1/video/chat/stream",
        "--model",
        "aurateam/AURA",
        "--video",
        FULL_VIDEO,
        "--burst-interval",
        "0",
        "--fps",
        "2",
        "--no-evs",
    ]
    for sched in FULL_AUDIO_SCHEDULE:
        cmd.extend(["--audio-schedule", sched])
    return cmd


def _save_summary(log_dir: Path, state: SoakState) -> None:
    (log_dir / "summary.json").write_text(
        json.dumps(asdict(state), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_final_report(log_dir: Path, state: SoakState, deadline_hit: bool) -> None:
    lines = [
        "# AURA Streaming Soak — Final Report",
        "",
        f"- Started: see soak.log",
        f"- Status: **{state.status}**",
        f"- Consecutive passes: **{state.consecutive_pass}**",
        f"- Total full runs: {state.total_full_runs}",
        f"- Server boots: {state.server_boots}",
        f"- Deadline hit (12h): {deadline_hit}",
        "",
    ]
    if state.failure_reason:
        lines.append(f"- Last failure: {state.failure_reason}")
        lines.append("")
    if state.runs:
        lines.append("## Recent runs")
        lines.append("")
        for rec in state.runs[-15:]:
            lines.append(
                f"- {rec.kind} #{rec.run_id}: {'PASS' if rec.passed else 'FAIL'} "
                f"({rec.duration_s:.0f}s) {rec.reasons}"
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Log directory: `{log_dir}`",
            f"- Server log: `{log_dir / 'server.log'}`",
            f"- Failures: `{log_dir / 'failures'}`",
        ]
    )
    (log_dir / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_failure_bundle(
    log_dir: Path,
    label: str,
    demo_log: Path,
    server_log: Path,
    server_offset: int,
    reasons: list[str],
) -> None:
    fail_dir = log_dir / "failures" / label
    fail_dir.mkdir(parents=True, exist_ok=True)
    if demo_log.exists():
        (fail_dir / "demo.log").write_text(demo_log.read_text(encoding="utf-8"), encoding="utf-8")
    snippet = _read_server_log(server_log, server_offset)
    (fail_dir / "server_snippet.log").write_text(snippet, encoding="utf-8")
    (fail_dir / "validate_reasons.txt").write_text("\n".join(reasons) + "\n", encoding="utf-8")


def run_soak(target_consecutive: int, max_hours: float, health_timeout_s: float) -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = _REPO_ROOT / "logs" / f"aura_soak_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = log_dir / "failures"
    failures_dir.mkdir(exist_ok=True)

    state = SoakState()
    deadline = time.monotonic() + max_hours * 3600
    server_proc: subprocess.Popen[Any] | None = None
    server_log = log_dir / "server.log"
    deadline_hit = False

    def _shutdown_server() -> None:
        nonlocal server_proc
        if server_proc is not None and server_proc.poll() is None:
            server_proc.send_signal(signal.SIGTERM)
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        server_proc = None
        _kill_port(PORT)

    try:
        while state.consecutive_pass < target_consecutive and time.monotonic() < deadline:
            _shutdown_server()
            state.server_boots += 1
            _log(f"=== Server boot #{state.server_boots} ===", log_dir)

            with server_log.open("a", encoding="utf-8") as slog:
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "2,3")}
                server_proc = subprocess.Popen(
                    SERVER_CMD,
                    cwd=str(_REPO_ROOT),
                    env=env,
                    stdout=slog,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

            if not _wait_health(health_timeout_s, log_dir):
                state.failure_reason = "health check timeout"
                _log("health check failed; retrying", log_dir)
                _save_failure_bundle(
                    log_dir,
                    f"boot_{state.server_boots}_health",
                    log_dir / "warmup_run.log",
                    server_log,
                    0,
                    ["health check timeout"],
                )
                continue

            # Warmup
            warmup_log = log_dir / "warmup_run.log"
            server_offset = server_log.stat().st_size if server_log.exists() else 0
            t0 = time.monotonic()
            _log("running warmup demo", log_dir)
            rc, demo_out = _run_demo(_warmup_cmd(), warmup_log, recv_timeout=900.0)
            server_snip = _read_server_log(server_log, server_offset)
            warmup_result = validate_run(demo_out, server_snip, warmup=True)
            if rc != 0:
                warmup_result.passed = False
                warmup_result.reasons.append(f"demo exit code {rc}")
            rec = RunRecord(
                kind="warmup",
                run_id=state.server_boots,
                passed=warmup_result.passed,
                reasons=warmup_result.reasons,
                started_at=_utc_now(),
                ended_at=_utc_now(),
                duration_s=time.monotonic() - t0,
            )
            state.runs.append(rec)
            _save_summary(log_dir, state)
            if not warmup_result.passed:
                state.consecutive_pass = 0
                state.failure_reason = "; ".join(warmup_result.reasons)
                _log(f"warmup FAILED: {warmup_result.reasons}", log_dir)
                _save_failure_bundle(
                    log_dir,
                    f"boot_{state.server_boots}_warmup",
                    warmup_log,
                    server_log,
                    server_offset,
                    warmup_result.reasons,
                )
                continue
            _log("warmup PASSED", log_dir)

            # Full demos on same server
            while state.consecutive_pass < target_consecutive and time.monotonic() < deadline:
                state.total_full_runs += 1
                run_id = state.total_full_runs
                full_log = log_dir / f"full_run_{run_id:02d}.log"
                server_offset = server_log.stat().st_size if server_log.exists() else 0
                t0 = time.monotonic()
                _log(f"running full demo #{run_id} (consecutive={state.consecutive_pass})", log_dir)
                rc, demo_out = _run_demo(_full_cmd(), full_log, recv_timeout=7200.0)
                server_snip = _read_server_log(server_log, server_offset)
                full_result = validate_run(demo_out, server_snip, warmup=False)
                if rc != 0:
                    full_result.passed = False
                    full_result.reasons.append(f"demo exit code {rc}")
                rec = RunRecord(
                    kind="full",
                    run_id=run_id,
                    passed=full_result.passed,
                    reasons=full_result.reasons,
                    started_at=_utc_now(),
                    ended_at=_utc_now(),
                    duration_s=time.monotonic() - t0,
                )
                state.runs.append(rec)
                _save_summary(log_dir, state)

                if full_result.passed:
                    state.consecutive_pass += 1
                    _log(
                        f"full demo #{run_id} PASSED "
                        f"(consecutive={state.consecutive_pass}/{target_consecutive})",
                        log_dir,
                    )
                else:
                    state.consecutive_pass = 0
                    state.failure_reason = "; ".join(full_result.reasons)
                    _log(f"full demo #{run_id} FAILED: {full_result.reasons}", log_dir)
                    _save_failure_bundle(
                        log_dir,
                        f"full_run_{run_id:02d}",
                        full_log,
                        server_log,
                        server_offset,
                        full_result.reasons,
                    )
                    break

        if time.monotonic() >= deadline and state.consecutive_pass < target_consecutive:
            deadline_hit = True
            state.status = "timeout"
            _log("deadline reached", log_dir)
        elif state.consecutive_pass >= target_consecutive:
            state.status = "success"
            _log(f"SUCCESS: {target_consecutive} consecutive passes", log_dir)
        else:
            state.status = "stopped"

    finally:
        _shutdown_server()
        _write_final_report(log_dir, state, deadline_hit)
        _save_summary(log_dir, state)
        _log(f"soak finished: status={state.status} consecutive={state.consecutive_pass}", log_dir)

    return 0 if state.status == "success" else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="AURA streaming 12h soak")
    parser.add_argument("--target-consecutive", type=int, default=10)
    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument("--health-timeout", type=float, default=2400.0)
    args = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3")
    raise SystemExit(run_soak(args.target_consecutive, args.max_hours, args.health_timeout))


if __name__ == "__main__":
    main()
