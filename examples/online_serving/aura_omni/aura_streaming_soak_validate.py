# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validators for AURA streaming soak runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_TURN_SUMMARY_RE = re.compile(r"Session summary:\s*(\d+)\s+turn\(s\)")
_AURA_PROMPT_RE = re.compile(
    r"AURA turn prompt request_id=\S+\s+transcript=(.+?):\s*(\{.*\})\s*$",
)

_SERVER_FAIL_PATTERNS = (
    "Traceback",
    "AssertionError",
    "num_scheduled_tokens=-1",
    "Query processing failed",
    "Engine query failed",
)


@dataclass
class ValidationResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> None:
        self.passed = False
        self.reasons.append(reason)


def _parse_turn_count(demo_output: str) -> int | None:
    matches = _TURN_SUMMARY_RE.findall(demo_output)
    if not matches:
        return None
    return int(matches[-1])


def validate_demo_output(demo_output: str, *, warmup: bool = False) -> ValidationResult:
    result = ValidationResult(passed=True)
    min_turns = 1 if warmup else 3

    if "session.done" not in demo_output.lower() and "<<< session.done" not in demo_output:
        result.fail("missing session.done")
    if "<<< ERROR" in demo_output or "error " in demo_output.lower():
        if "<<< ERROR:" in demo_output:
            result.fail("demo reported ERROR")
    if "Timed out" in demo_output:
        result.fail("demo timed out waiting for session.done")

    turn_count = _parse_turn_count(demo_output)
    if turn_count is None:
        result.fail("missing Session summary turn count")
    elif turn_count < min_turns:
        result.fail(f"turn count {turn_count} < required {min_turns}")

    return result


def validate_server_log(server_log: str) -> ValidationResult:
    result = ValidationResult(passed=True)
    for pattern in _SERVER_FAIL_PATTERNS:
        if pattern in server_log:
            result.fail(f"server log contains {pattern!r}")
    return result


def _history_segment_before_final_user(prompt_text: str) -> str:
    """Return prompt body before the last user turn (history region)."""
    marker = "<|im_start|>user"
    last = prompt_text.rfind(marker)
    if last <= 0:
        return prompt_text
    return prompt_text[:last]


def _max_consecutive_assistants_without_user(text: str) -> int:
    tokens = re.split(r"(<\|im_start\|>user|<\|im_start\|>assistant)", text)
    max_run = 0
    current = 0
    for tok in tokens:
        if tok == "<|im_start|>assistant":
            current += 1
            max_run = max(max_run, current)
        elif tok == "<|im_start|>user":
            current = 0
    return max_run


def _parse_aura_prompt_lines(server_log: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in server_log.splitlines():
        match = _AURA_PROMPT_RE.search(line)
        if not match:
            continue
        transcript_repr = match.group(1).strip()
        payload_raw = match.group(2)
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            continue
        transcript = transcript_repr.strip("'\"")
        entries.append(
            {
                "transcript": transcript,
                "prompt_text": payload.get("prompt_text", ""),
                "videos": payload.get("videos", []),
            }
        )
    return entries


def validate_aura_prompt_logs(server_log: str) -> ValidationResult:
    result = ValidationResult(passed=True)
    entries = _parse_aura_prompt_lines(server_log)
    if not entries:
        result.fail("no AURA turn prompt lines found in server log (check AURA worker log capture)")
        return result

    for idx, entry in enumerate(entries, start=1):
        transcript = entry["transcript"]
        if "<asr_text>" in transcript:
            result.fail(f"turn {idx}: transcript still contains <asr_text>")
        if transcript.lower().startswith("language chinese"):
            result.fail(f"turn {idx}: transcript has uncleaned language prefix")

        history = _history_segment_before_final_user(entry["prompt_text"])
        max_asst = _max_consecutive_assistants_without_user(history)
        if max_asst >= 3:
            result.fail(
                f"turn {idx}: {max_asst} consecutive assistant blocks without user "
                "(possible SessionHistory persistence bug)"
            )

        if idx >= 3:
            videos = entry.get("videos") or []
            min_videos = min(idx - 1, 2)
            if len(videos) < min_videos:
                result.fail(f"turn {idx}: videos length {len(videos)} < expected {min_videos}")

    return result


def validate_run(
    demo_output: str,
    server_log: str,
    *,
    warmup: bool = False,
) -> ValidationResult:
    combined = ValidationResult(passed=True)
    for part in (
        validate_demo_output(demo_output, warmup=warmup),
        validate_server_log(server_log),
    ):
        if not part.passed:
            combined.passed = False
        combined.reasons.extend(part.reasons)

    if not warmup:
        prompt_result = validate_aura_prompt_logs(server_log)
        if not prompt_result.passed:
            combined.passed = False
        combined.reasons.extend(prompt_result.reasons)

    return combined
