import argparse
import os

from vllm.benchmarks.serve import add_cli_args

from vllm_omni.benchmarks.serve import main
from vllm_omni.entrypoints.cli.benchmark.base import OmniBenchmarkSubcommandBase


def add_daily_omni_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments specific to Daily-Omni dataset.

    This function should be called by the CLI entrypoint to add additional
    arguments for daily-omni benchmark support.

    Args:
        parser: The ArgumentParser instance to extend
    """
    # Daily-Omni specific arguments
    daily_omni_group = parser.add_argument_group("Daily-Omni Dataset Options")

    daily_omni_group.add_argument(
        "--daily-omni-qa-json",
        type=str,
        default=None,
        help="Path to local upstream qa.json. When set, QA rows are read from this file and "
        "the HuggingFace dataset is not loaded (no network). Use with --daily-omni-video-dir "
        "for fully offline runs. --dataset-path / Hub split flags are then ignored for QA loading.",
    )
    daily_omni_group.add_argument(
        "--daily-omni-video-dir",
        type=str,
        default=None,
        help="Root directory of extracted Daily-Omni videos (contents of Videos.tar: "
        "each video_id in its own subdir with {video_id}_video.mp4). "
        "If omitted, Videos.tar is downloaded from the Hugging Face dataset repo on first multimodal "
        "request. "
        "When using file URLs, you MUST start the vLLM server with "
        "--allowed-local-media-path set to this same directory (or a parent), "
        "otherwise requests fail with 'Cannot load local files without "
        "--allowed-local-media-path'.",
    )
    daily_omni_group.add_argument(
        "--daily-omni-inline-local-video",
        action="store_true",
        default=False,
        help="For local videos only: embed MP4 as base64 data URLs in benchmark "
        "requests so the server does not need --allowed-local-media-path. "
        "Increases request size and client memory; use for small --num-prompts. "
        "When using --daily-omni-input-mode audio or all, local WAV files are "
        "embedded the same way.",
    )
    daily_omni_group.add_argument(
        "--daily-omni-input-mode",
        type=str,
        choices=["all", "visual", "audio"],
        default="all",
        help="Daily-Omni input protocol (mirrors upstream Lliar-liar/Daily-Omni "
        "--input_mode). 'visual': video only (default). 'audio': WAV only, "
        "requires {video_id}/{video_id}_audio.wav under --daily-omni-video-dir. "
        "'all': video + WAV together. Sets mm_processor_kwargs.use_audio_in_video=false "
        "and matches official separate video/audio streams.",
    )
    daily_omni_group.add_argument(
        "--daily-omni-save-eval-items",
        action="store_true",
        default=False,
        help="Include per-request Daily-Omni accuracy rows (gold/predicted/correct) "
        "in the saved JSON under key daily_omni_eval_items. "
        "Alternatively set env DAILY_OMNI_SAVE_EVAL_ITEMS=1.",
    )

    # Note: --dataset-name daily-omni via get_samples patch; use either Hub (--dataset-path
    # liarliar/Daily-Omni) or local --daily-omni-qa-json (offline).


def add_seed_tts_cli_args(parser: argparse.ArgumentParser) -> None:
    """CLI for Seed-TTS zero-shot TTS benchmark (``--dataset-name seed-tts``)."""
    g = parser.add_argument_group("Seed-TTS Dataset Options")
    g.add_argument(
        "--seed-tts-locale",
        type=str,
        choices=["en", "zh"],
        default="en",
        help="Which Seed-TTS split to load: en/meta.lst or zh/meta.lst under the dataset root.",
    )
    g.add_argument(
        "--seed-tts-root",
        type=str,
        default=None,
        help="Override root directory that contains en/ and zh/ (meta.lst + prompt-wavs). "
        "If set, --dataset-path can still name the HF repo for logging; this path is used for files.",
    )
    g.add_argument(
        "--seed-tts-file-ref-audio",
        action="store_true",
        default=False,
        help="Send ref_audio as file:// URIs (smaller HTTP bodies). Requires the API server "
        "to be started with --allowed-local-media-path covering the Seed-TTS dataset root. "
        "Default is inline data:audio/wav;base64 so Qwen3-TTS works without that flag.",
    )
    g.add_argument(
        "--seed-tts-inline-ref-audio",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    g.add_argument(
        "--seed-tts-system-prompt",
        type=str,
        default=None,
        help="Override chat system message for --backend openai-chat-omni (Qwen3-Omni TTS). "
        "Default follows official Qwen3-Omni identity + zero-shot voice-clone instructions.",
    )
    g.add_argument(
        "--seed-tts-wer-eval",
        action="store_true",
        default=False,
        help="Keep synthesized audio as 24 kHz mono PCM for WER (works with "
        "--backend openai-audio-speech or openai-chat-omni). Scoring follows "
        "zhaochenyang20/seed-tts-eval (Whisper-large-v3 / Paraformer-zh + jiwer). "
        "Sets SEED_TTS_WER_EVAL=1. Install: pip install 'vllm-omni[dev]'. "
        "Optional: SEED_TTS_EVAL_DEVICE, SEED_TTS_HF_WHISPER_MODEL.",
    )
    g.add_argument(
        "--seed-tts-wer-save-items",
        action="store_true",
        default=False,
        help="Include per-utterance ASR rows in the saved JSON under key seed_tts_wer_eval_items. "
        "Or set SEED_TTS_WER_SAVE_ITEMS=1.",
    )


def add_omniinteract_cli_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("OmniInteract Dataset Options")
    g.add_argument(
        "--omniinteract-root",
        type=str,
        default=None,
        help="Local OmniInteract extracted data root (contains 1q1a/1q1a_math/1qna, or a parent with data/). "
        "If omitted, benchmark downloads data.tar.gz from --dataset-path/--hf-name (default lucky-lance/OmniInteract).",
    )
    g.add_argument(
        "--omniinteract-subsets",
        type=str,
        default="1q1a,1q1a_math,1qna",
        help="Comma-separated subsets to evaluate, e.g. '1q1a,1q1a_math' or '1qna'.",
    )
    g.add_argument(
        "--omniinteract-inline-local-video",
        action="store_true",
        default=False,
        help="Embed local MP4 as data:video URLs instead of file://. Useful when server lacks "
        "--allowed-local-media-path; increases request body size.",
    )
    g.add_argument(
        "--omniinteract-input-mode",
        type=str,
        choices=["video", "aura"],
        default="video",
        help=(
            "OmniInteract request protocol. 'video' sends subvideo_url + text question with "
            "mm_processor_kwargs.use_audio_in_video=true. 'aura' sends paired audio_url + "
            "subvideo_url and AURA/TTS extra_body fields for the ASR -> AURA -> TTS pipeline. "
            "Generate subvideos/ and audios/ under 1q1a before benchmarking."
        ),
    )
    g.add_argument(
        "--omniinteract-aura-tts-task-type",
        type=str,
        choices=["Base", "CustomVoice"],
        default="Base",
        help=(
            "TTS task type for OmniInteract AURA mode. Base requires "
            "--omniinteract-aura-tts-ref-audio and --omniinteract-aura-tts-ref-text."
        ),
    )
    g.add_argument(
        "--omniinteract-aura-tts-language",
        type=str,
        default="Chinese",
        help="TTS language passed to OmniInteract AURA TTS.",
    )
    g.add_argument(
        "--omniinteract-aura-tts-speaker",
        type=str,
        default=None,
        help="TTS speaker passed to OmniInteract AURA CustomVoice mode.",
    )
    g.add_argument(
        "--omniinteract-aura-tts-ref-audio",
        type=str,
        default=None,
        help="Reference audio path/URL for OmniInteract AURA Base TTS mode.",
    )
    g.add_argument(
        "--omniinteract-aura-tts-ref-text",
        type=str,
        default=None,
        help="Reference text transcript for OmniInteract AURA Base TTS mode.",
    )
    g.add_argument(
        "--omniinteract-eval",
        action="store_true",
        default=False,
        help="Compute OmniInteract QA metrics (IA-QTF1/IDS/NCCS). Disabled by default; "
        "use for accuracy runs. Perf-only runs collect serving metrics only.",
    )
    g.add_argument(
        "--omniinteract-save-eval-items",
        action="store_true",
        default=False,
        help="When --omniinteract-eval is set, include per-request OmniInteract eval rows in "
        "result JSON as omniinteract_eval_items. Alternatively set OMNIINTERACT_SAVE_EVAL_ITEMS=1.",
    )


def add_omni_benchmark_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add vLLM-Omni specific serving benchmark options."""
    group = parser.add_argument_group("vLLM-Omni Multi-stage Benchmark Options")
    group.add_argument(
        "--print-stage",
        action="store_true",
        default=False,
        help=(
            "Print per-stage benchmark metrics for --omni serving when stage metrics are returned by the server. "
            "Disabled by default. The latency sections follow --percentile-metrics by modality: "
            "ttft/tpot/itl control text stages, ttfc/tpoc/icl control internal stream stages, "
            "and tpop controls both text TPOP and internal stream TPOP."
        ),
    )
    group.add_argument(
        "--image-edits-bot-task",
        type=str,
        default="think",
        help=(
            "Default bot_task form field for --backend openai-image-edits-omni "
            '(/v1/images/edits). Use --extra-body \'{"bot_task":"..."}\' to override per run.'
        ),
    )


_OMNI_BENCH_DATASET_CHOICES = (
    "daily-omni",
    "omniinteract",
    "seed-tts",
    "seed-tts-text",
    "seed-tts-design",
    "ttsd",
    "sound-effect",
)


def _extend_omni_dataset_name_choices(parser: argparse.ArgumentParser) -> None:
    """Append omni benchmark dataset names to --dataset-name choices.

    TrackingArgumentParser keeps a shadow parser for explicit-arg tracking; both
    parsers must list the same choices or parse_args rejects valid omni values.
    """
    parsers = [parser]
    shadow = getattr(parser, "_shadow", None)
    if shadow is not None:
        parsers.append(shadow)

    for p in parsers:
        for action in p._actions:
            if action.dest == "dataset_name" and action.choices is not None:
                extra = [c for c in _OMNI_BENCH_DATASET_CHOICES if c not in action.choices]
                if extra:
                    action.choices = list(action.choices) + extra


class OmniBenchmarkServingSubcommand(OmniBenchmarkSubcommandBase):
    """The `serve` subcommand for vllm bench."""

    name = "serve"
    help = "Benchmark the online serving throughput. Supports Daily-Omni and Seed-TTS datasets."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

        # Add Daily-Omni specific arguments
        add_daily_omni_cli_args(parser)
        add_omniinteract_cli_args(parser)
        add_seed_tts_cli_args(parser)
        add_omni_benchmark_cli_args(parser)

        for action in parser._actions:
            if action.dest == "dataset_name" and action.choices is not None:
                extra = [
                    c
                    for c in (
                        "daily-omni",
                        "omniinteract",
                        "seed-tts",
                        "seed-tts-text",
                        "seed-tts-design",
                        "ttsd",
                        "sound-effect",
                    )
                    if c not in action.choices
                ]
                if extra:
                    action.choices = list(action.choices) + extra
            if action.dest == "backend" and action.choices is not None:
                extra = [c for c in ("openai-image-edits-omni",) if c not in action.choices]
                if extra:
                    action.choices = list(action.choices) + extra
        _extend_omni_dataset_name_choices(parser)

        # Update help messages for omni-specific features
        for action in parser._actions:
            if action.dest == "percentile_metrics":
                action.help = (
                    "Comma-separated list of selected metrics to report percentiles. "
                    'For text metrics, "ttft", "tpot", and "itl" affect the global benchmark and text '
                    'stage metrics. "tpop" also requests text TPOT/TPOP globally and per stage, and internal '
                    'stream TPOP. "ttfc", "tpoc", and "icl" only affect internal stream stage metrics. '
                    'Audio metrics include "audio_ttfp", "audio_rtf", "audio_duration", and "audio_underrun".'
                )
            if action.dest == "random_mm_limit_mm_per_prompt":
                action.help = (
                    "Per-modality hard caps for items attached per request, e.g. "
                    '\'{"image": 3, "video": 0, "audio": 1}\'. The sampled per-request item '
                    "count is clamped to the sum of these limits. When a modality "
                    "reaches its cap, its buckets are excluded and probabilities are "
                    "renormalized."
                )
            if action.dest == "random_mm_bucket_config":
                action.help = (
                    "The bucket config is a dictionary mapping a multimodal item"
                    "sampling configuration to a probability."
                    "Currently allows for 3 modalities: audio, images and videos. "
                    "A bucket key is a tuple of (height, width, num_frames)"
                    "The value is the probability of sampling that specific item. "
                    "Example: "
                    "--random-mm-bucket-config "
                    "{(256, 256, 1): 0.5, (720, 1280, 16): 0.4, (0, 1, 5): 0.10} "
                    "First item: images with resolution 256x256 w.p. 0.5"
                    "Second item: videos with resolution 720x1280 and 16 frames "
                    "Third item: audios with 1s duration and 5 channels w.p. 0.1"
                    "OBS.: If the probabilities do not sum to 1, they are normalized."
                )

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        if getattr(args, "daily_omni_save_eval_items", False):
            os.environ["DAILY_OMNI_SAVE_EVAL_ITEMS"] = "1"
        if getattr(args, "omniinteract_eval", False):
            os.environ["OMNIINTERACT_EVAL"] = "1"
        if getattr(args, "omniinteract_save_eval_items", False):
            os.environ["OMNIINTERACT_SAVE_EVAL_ITEMS"] = "1"
        if getattr(args, "seed_tts_wer_eval", False):
            os.environ["SEED_TTS_WER_EVAL"] = "1"
        if getattr(args, "seed_tts_wer_save_items", False):
            os.environ["SEED_TTS_WER_SAVE_ITEMS"] = "1"
        image_edits_bot_task = getattr(args, "image_edits_bot_task", None)
        if image_edits_bot_task is not None:
            extra_body = dict(getattr(args, "extra_body", None) or {})
            extra_body.setdefault("bot_task", image_edits_bot_task)
            args.extra_body = extra_body
        main(args)
