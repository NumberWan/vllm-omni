"""Detect non-semantic residue in AURA OmniInteract streaming outputs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{1,}")
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_CODE_LATEX_RE = re.compile(
    r"```|\\(?:section|subsection|begin|end|IEEEPARstart|documentclass)\b",
    re.IGNORECASE,
)
_JSON_RE = re.compile(
    r"(?:^|\n)\s*[\[{]\s*(?:[\"'][^\"']+[\"']\s*:|[\"'][A-Za-z_])",
    re.DOTALL,
)
_CITATION_RE = re.compile(
    r"\b(?:doi|et\s+al|vol\.?|pp\.?|references?|bibliography)\b"
    r"|\b\d{2,4}\s+[A-Z][a-z]{2};\d+\(\d+\):\d+"
    r"|\[[0-9]{1,3}\]",
    re.IGNORECASE,
)
_PROGRAM_RE = re.compile(
    r"\b(?:def|class|import|from|return|pytest|pip install|npm|function)\b"
    r"|(?:^|\n)\s*(?:for|while|if)\s+.+:",
    re.IGNORECASE,
)
_COORDINATE_RE = re.compile(
    r"^\s*(?:\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)\s*,?\s*)+$"
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_PUNCTUATION_ONLY_RE = re.compile(r"^[\s，。！？、,.!?;；:：…—~～\-]+$")


@dataclass(frozen=True)
class QualityFinding:
    code: str
    detail: str


def _is_effectively_silent(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped == "<|silent|>":
        return True
    natural = _THINK_RE.sub("", stripped).strip()
    return not natural or bool(_PUNCTUATION_ONLY_RE.fullmatch(natural))


def _repeated_long_line(text: str) -> bool:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if len(line) >= 24]
    return any(count > 1 for count in Counter(lines).values())


def classify_residue(text: str) -> list[QualityFinding]:
    """Return deterministic quality findings for one visible spoken turn."""

    if not isinstance(text, str) or _is_effectively_silent(text):
        return []
    stripped = text.strip()
    findings: list[QualityFinding] = []
    cjk_count = len(_CJK_RE.findall(stripped))
    latin_words = _LATIN_WORD_RE.findall(stripped)
    latin_chars = sum(len(word) for word in latin_words)

    if _CODE_LATEX_RE.search(stripped):
        findings.append(QualityFinding("code_or_latex", "code fence or LaTeX command"))
    if _JSON_RE.search(stripped):
        findings.append(QualityFinding("json_blob", "JSON-like object/array"))
    if _CITATION_RE.search(stripped):
        findings.append(QualityFinding("paper_citation", "paper/reference-like citation"))
    if _PROGRAM_RE.search(stripped) and len(latin_words) >= 4:
        findings.append(QualityFinding("program_text", "programming/instruction residue"))
    if _COORDINATE_RE.fullmatch(stripped):
        findings.append(QualityFinding("coordinate_fragment", "coordinate-only fragment"))
    if _KANA_RE.search(stripped):
        findings.append(QualityFinding("unexpected_kana", "Japanese kana in Chinese benchmark"))
    if len(latin_words) >= 8 and (cjk_count < 8 or latin_chars > cjk_count * 2):
        findings.append(
            QualityFinding(
                "english_paragraph",
                f"English-dominant paragraph ({len(latin_words)} words, {cjk_count} CJK chars)",
            )
        )
    if _repeated_long_line(stripped):
        findings.append(QualityFinding("repeated_ramble", "repeated long line"))
    return findings


def _iter_turns(result: dict[str, Any]):
    for request in result.get("per_requests") or []:
        video_id = str(request.get("video_id") or Path(str(request.get("video_path") or "")).stem)
        turns = request.get("streaming_turns") or request.get("streaming_chunks") or []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            text = str(turn.get("text") or "")
            if _is_effectively_silent(text):
                continue
            yield video_id, turn, text


def evaluate_result(result: dict[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    spoken = 0
    by_code: Counter[str] = Counter()
    for video_id, turn, text in _iter_turns(result):
        spoken += 1
        findings = classify_residue(text)
        if not findings:
            continue
        for finding in findings:
            by_code[finding.code] += 1
        violations.append(
            {
                "video_id": video_id,
                "timestamp": turn.get("timestamp"),
                "response_index": turn.get("response_index"),
                "codes": [finding.code for finding in findings],
                "details": [finding.detail for finding in findings],
                "text": text,
            }
        )
    return {
        "spoken_turns": spoken,
        "residue_turns": len(violations),
        "residue_by_code": dict(sorted(by_code.items())),
        "omniinteract_evaluated": int(result.get("omniinteract_evaluated") or 0),
        "omniinteract_ia_qtf1": result.get("omniinteract_ia_qtf1"),
        "violations": violations,
        "passed": not violations,
    }


def _render_markdown(report: dict[str, Any], source: Path) -> str:
    lines = [
        "# AURA OmniInteract quality gate",
        "",
        f"- source: `{source}`",
        f"- passed: **{report['passed']}**",
        f"- spoken turns: **{report['spoken_turns']}**",
        f"- residue turns: **{report['residue_turns']}**",
        f"- evaluated QA slots: **{report['omniinteract_evaluated']}**",
        f"- IA-QTF1: **{report['omniinteract_ia_qtf1']}**",
        "",
    ]
    for item in report["violations"]:
        lines.extend(
            [
                f"## {item['video_id']} @ {item['timestamp']}",
                "",
                f"Codes: `{', '.join(item['codes'])}`",
                "",
                "```text",
                item["text"],
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--max-residue-turns", type=int, default=0)
    parser.add_argument("--min-spoken-turns", type=int, default=0)
    parser.add_argument("--min-evaluated-slots", type=int, default=0)
    parser.add_argument("--min-ia-qtf1", type=float)
    args = parser.parse_args()

    result = json.loads(args.result_json.read_text(encoding="utf-8"))
    report = evaluate_result(result)
    report["passed"] = bool(
        report["residue_turns"] <= args.max_residue_turns
        and report["spoken_turns"] >= args.min_spoken_turns
        and report["omniinteract_evaluated"] >= args.min_evaluated_slots
        and (
            args.min_ia_qtf1 is None
            or (
                report["omniinteract_ia_qtf1"] is not None
                and float(report["omniinteract_ia_qtf1"]) >= args.min_ia_qtf1
            )
        )
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(
            _render_markdown(report, args.result_json),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
