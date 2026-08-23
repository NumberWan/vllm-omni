from vllm_omni.benchmarks.omniinteract_quality_gate import (
    classify_residue,
    evaluate_result,
)


def _codes(text: str) -> set[str]:
    return {finding.code for finding in classify_residue(text)}


def test_quality_gate_accepts_short_chinese_and_named_entities():
    assert _codes("没问题，等《古韵》出现时我马上提醒你。") == set()
    assert _codes("AURA 模型目前运行正常，GPU 使用率很低。") == set()
    assert _codes("<|silent|>") == set()
    assert _codes("<think>\n\n</think>") == set()


def test_quality_gate_detects_known_aura_v2_residue_shapes():
    assert "json_blob" in _codes('```json\n{"Question": "density of air"}')
    assert "code_or_latex" in _codes(r"\section{Introduction}\n\IEEEPARstart{A}{utomated} driving")
    assert "english_paragraph" in _codes(
        "The following is a list of the most important works in the field of economics."
    )
    assert "paper_citation" in _codes("The blue thermos is on the desk. 197 Oct;96(10):3241–3250.")
    assert "coordinate_fragment" in _codes("(0,0.0),(9,510.0)")
    assert "unexpected_kana" in _codes("らo")
    assert "program_text" in _codes(
        "Install Dependencies: pip install package, then run the tests in the file."
    )


def test_evaluate_result_counts_spoken_violations_and_metrics():
    report = evaluate_result(
        {
            "omniinteract_evaluated": 12,
            "omniinteract_ia_qtf1": 0.2,
            "per_requests": [
                {
                    "video_id": "0002",
                    "streaming_turns": [
                        {"text": "<|silent|>", "is_silent": True},
                        {"text": "好的，我会提醒你。", "is_silent": False},
                        {
                            "text": r"\section{Introduction}",
                            "is_silent": False,
                            "timestamp": [1.0, 2.0],
                        },
                    ],
                }
            ],
        }
    )

    assert report["spoken_turns"] == 2
    assert report["residue_turns"] == 1
    assert report["residue_by_code"] == {"code_or_latex": 1}
    assert report["omniinteract_evaluated"] == 12
    assert report["passed"] is False
