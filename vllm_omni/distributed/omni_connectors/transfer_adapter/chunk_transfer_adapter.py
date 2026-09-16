# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from typing import Any

import torch
from vllm.v1.metrics.stats import PrefillStats
from vllm.v1.request import Request, RequestStatus

from vllm_omni.data_entry_keys import MetaStruct, OmniPayloadStruct, unflatten_payload

from ..adapter import construct_next_stage_streaming_input_prompt
from ..factory import OmniConnectorFactory
from ..utils.config import ConnectorSpec, stage_receives_chunks
from ..utils.logging import get_connector_logger
from .base import OmniTransferAdapterBase

logger = get_connector_logger(__name__)


def _mark_async_chunk_stage_ready(request: Any) -> float:
    """Stamp stage-local ready time once (first usable upstream chunk).

    async_chunk prewarms Stage-1+ at request start. TTFx for those stages must
    start when the upstream chunk actually arrives, not at prewarm — otherwise
    idle wait for ASR / prior stages is billed to Stage-N.
    """
    existing = getattr(request, "_omni_stage_ready_ts", None)
    if existing is not None:
        return float(existing)
    ready_ts = time.time()
    request._omni_stage_ready_ts = ready_ts
    request.arrival_time = ready_ts
    return ready_ts


def _replace_request_prompt_token_ids(request: Any, prompt_token_ids: list[int]) -> bool:
    if not prompt_token_ids:
        return False
    if int(getattr(request, "num_computed_tokens", 0) or 0) > 0:
        logger.warning(
            "Received prompt_token_ids for req=%s after prefill started; keeping existing prompt_len=%s",
            getattr(request, "request_id", None),
            len(getattr(request, "prompt_token_ids", []) or []),
        )
        return False
    request.prompt_token_ids = list(prompt_token_ids)
    request.num_prompt_tokens = len(prompt_token_ids)
    if hasattr(request, "_output_token_ids"):
        request._output_token_ids.clear()
    if hasattr(request, "_all_token_ids"):
        request._all_token_ids.clear()
        request._all_token_ids.extend(prompt_token_ids)
    if hasattr(request, "block_hashes"):
        request.block_hashes.clear()
    if hasattr(request, "update_block_hashes"):
        request.update_block_hashes()
    return True


def _coerce_positive_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        for item in value:
            coerced = _coerce_positive_int(item)
            if coerced > 0:
                return coerced
        return 0
    if hasattr(value, "item"):
        value = value.item()
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return 0
    return coerced if coerced > 0 else 0


def _extract_max_new_tokens_from_payload(payload_data: Any) -> int:
    """Read Talker length cap from flat or nested async_chunk payloads."""
    if not isinstance(payload_data, dict):
        return 0
    top = _coerce_positive_int(payload_data.get("max_new_tokens") or payload_data.get("tts_max_new_tokens"))
    if top > 0:
        return top
    additional = payload_data.get("additional_information")
    if isinstance(additional, dict):
        return _coerce_positive_int(additional.get("max_new_tokens") or additional.get("tts_max_new_tokens"))
    return 0


def _apply_max_new_tokens_from_payload(request: Any, payload_data: Any) -> int:
    """Clamp prewarmed Talker ``max_tokens`` using payload ``max_new_tokens``.

    async_chunk prewarms Stage2 with deploy default ``max_tokens`` (often 4096).
    The real TTS payload later arrives over SharedMemory with a text-proportional
    ``max_new_tokens`` cap, but without this clamp the Talker can babble for many
    seconds when EOS is late (e.g. ``你好。`` → 200 frames / ~16s).

    Call only at segment start (computed == prompt, no generated tokens).
    Mid-decode apply is a protocol bug: queue the next sentence instead of
    clamping below/around live ``num_computed``.
    """
    max_new_tokens = _extract_max_new_tokens_from_payload(payload_data)
    if max_new_tokens <= 0:
        return 0

    prompt_len = int(getattr(request, "num_prompt_tokens", 0) or 0)
    computed = int(getattr(request, "num_computed_tokens", 0) or 0)
    already_generated = max(0, computed - prompt_len) if prompt_len > 0 else 0
    if already_generated > 0:
        raise RuntimeError(
            f"Refuse max_new_tokens={max_new_tokens} mid-decode "
            f"(computed={computed} prompt_len={prompt_len} already_generated={already_generated}). "
            "Queue the next sentence until Talker WAITING; do not clamp a live budget."
        )

    sampling_params = getattr(request, "sampling_params", None)
    if sampling_params is not None and getattr(sampling_params, "max_tokens", None) is not None:
        sampling_params.max_tokens = min(int(sampling_params.max_tokens), max_new_tokens)

    current = getattr(request, "max_tokens", None)
    if current is None:
        effective = max_new_tokens
        request.max_tokens = effective
    else:
        effective = min(int(current), max_new_tokens)
        request.max_tokens = effective

    if effective != current:
        logger.info(
            "[async_chunk] req=%s applied max_new_tokens=%d (was max_tokens=%s)",
            getattr(request, "request_id", None),
            effective,
            current,
        )
    return effective


def _meta_to_dict(meta: Any) -> dict[str, Any]:
    """Coerce payload meta (dict / Mapping / msgspec Struct) to a plain dict."""
    if meta is None:
        return {}
    if isinstance(meta, dict):
        return dict(meta)
    if isinstance(meta, Mapping):
        return dict(meta)
    try:
        import msgspec

        return dict(msgspec.structs.asdict(meta))
    except Exception:
        pass
    out: dict[str, Any] = {}
    for key in (
        "finished",
        "is_segment_finished",
        "next_stage_prompt_len",
        "replace_streaming_prompt",
        "codec_streaming",
    ):
        if hasattr(meta, key):
            out[key] = getattr(meta, key)
    return out


def _payload_has_talker_conditioning(payload_data: Any) -> bool:
    """True if payload can start/continue Talker (text or precomputed ids).

    Empty Stage1 finish sentinels carry finished=True with no text / prompt ids.
    Applying those as new_segment wipes live TTS conditioning and crashes
    Qwen3-TTS decode with "Missing Qwen3-TTS text conditioning".
    """
    if not isinstance(payload_data, Mapping):
        return False
    text_list = payload_data.get("text")
    if isinstance(text_list, list) and bool(text_list) and bool(text_list[0]):
        return True
    additional = payload_data.get("additional_information")
    if isinstance(additional, Mapping):
        text_list = additional.get("text")
        if isinstance(text_list, list) and bool(text_list) and bool(text_list[0]):
            return True
        if additional.get("precomputed_text_id") is not None:
            return True
    if payload_data.get("precomputed_text_id") is not None:
        return True
    hs = payload_data.get("hidden_states")
    if isinstance(hs, Mapping):
        tail = hs.get("trailing_text")
        if hasattr(tail, "numel") and int(tail.numel()) > 0:
            return True
        last = hs.get("last")
        if hasattr(last, "numel") and int(last.numel()) > 0:
            return True
    prompt_ids = payload_data.get("prompt_token_ids")
    if isinstance(prompt_ids, list) and prompt_ids:
        return True
    ids = payload_data.get("ids")
    if isinstance(ids, Mapping):
        prompt_ids = ids.get("prompt")
        if isinstance(prompt_ids, list) and prompt_ids:
            return True
    return False


class OmniChunkTransferAdapter(OmniTransferAdapterBase):
    """Chunk-level transfer adapter for Omni connector pipelines.

    This class coordinates per-request chunk exchange between adjacent stages,
    and implements asynchronous get/put of chunks via background threads.
    It tracks per-request chunk indices for put/get, and accumulates
    payloads across chunks (concatenating tensors/lists in AR mode). It also
    caches prompt token ids and additional information for scheduler use.

    Scheduler integration is handled via WAITING_FOR_CHUNK transitions:
    requests are moved to waiting for chunk deque while polling, then restored
    to waiting/running queues once a chunk arrives. The requests will finish
    loading chunk util detecting the payload "finished" flag.

    The base class owns background recv/save loops; load/save only enqueue
    work and return immediately.
    """

    def __init__(self, vllm_config: Any):
        model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.scheduler_max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        active_stream_window = int(getattr(model_config, "active_stream_window", 0) or 0)
        model_max_num_seqs = int(getattr(model_config, "max_num_seqs", self.scheduler_max_num_seqs) or 0)
        if model_max_num_seqs <= 0:
            model_max_num_seqs = self.scheduler_max_num_seqs
        self._active_window = min(active_stream_window, model_max_num_seqs) if active_stream_window > 0 else 0
        if self._active_window > 0:
            logger.info(
                "Bounded active-stream window enabled: K=%d. "
                "Multi-replica deployments require sticky per-stream routing across Stage 1 "
                "replicas (each replica owns an independent active-set; without sticky routing, "
                "a stream can be active on one replica and non-active on another and both will "
                "race to evict it).",
                self._active_window,
            )
        self.connector = self.create_connector(model_config)
        self.receives_chunks = stage_receives_chunks(model_config)
        super().__init__(model_config)
        self.model_config = model_config
        self.model_mode = getattr(model_config, "worker_type", None) or "ar"
        # Wired by StageEngineCoreProc after EngineCore builds mm_receiver_cache.
        # Late-attached async_chunk mm_features must go through this cache so the
        # GPU worker can rehydrate MultiModalFeatureSpec.data from SHM.
        self.mm_receiver_cache: Any | None = None
        # State specific to Chunk management
        self.custom_process_next_stage_input_func: Callable[..., OmniPayloadStruct | dict[str, Any] | None] | None = (
            None
        )
        custom_process_next_stage_input_func = getattr(model_config, "custom_process_next_stage_input_func", None)
        if custom_process_next_stage_input_func:
            module_path, func_name = custom_process_next_stage_input_func.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.custom_process_next_stage_input_func = getattr(module, func_name)
        # mapping for request id and chunk id
        self.put_req_chunk: dict[str, int] = defaultdict(int)
        self.get_req_chunk: dict[str, int] = defaultdict(int)
        # Segment-local chunk counter: incremented alongside put_req_chunk
        # but popped at segment boundaries (unlike put_req_chunk which is
        # request-global for connector key continuity).
        self.ramp_chunk_count: dict[str, int] = defaultdict(int)
        self.upstream_exhausted_requests: set[str] = set()
        # Mid-decode Stage1 sentences parked until the current Talker segment
        # stops. Drained when the scheduler rearms the connector (WAITING).
        self._pending_upstream_payloads: dict[str, deque[dict[str, Any]]] = {}
        # Empty Stage1 finish sentinels that exhausted upstream without opening
        # a new Talker segment. Scheduler must finish these (not re-schedule
        # decode with stale computed >> prompt_len).
        self._terminal_empty_finish_reqs: set[str] = set()
        self.segment_finished_requests: set[str] = set()
        self.request_payload = {}
        self.code_prompt_token_ids: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.request_ids_mapping: dict[str, str] = {}

        self.waiting_for_chunk_waiting_requests: deque[Any] = deque()
        self.waiting_for_chunk_running_requests: deque[Any] = deque()
        self.requests_with_ready_chunks = set()
        self.requests_origin_status = {}
        self._active_streams: dict[str, Any] = {}
        # Private hold-queue for non-active running requests. Restored to
        # running_queue inside restore_queues(). Avoids calling
        # waiting_queue.prepend_requests mid-step, which trips vllm's
        # per-step LogitsProcessor invariant
        # ("Cannot register new removed request after self.removed has
        #   been read").
        self._held_non_active: deque[Any] = deque()
        self.requests_num_chunks_sent: dict[str, int] = defaultdict(int)
        self._pending_streaming_prefills: dict[str, dict] = {}

    @staticmethod
    def _is_truthy_scalar(value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            return value.numel() == 1 and bool(value.item())
        return bool(value) if value is not None else False

    @classmethod
    def _payload_terminal_flags(cls, payload_data: Any) -> tuple[bool, bool]:
        """Read finished / is_segment_finished from nested, flat, or struct meta.

        SharedMemory round-trips keep nested ``meta.finished``. Some worker
        paths flatten to ``meta.finished`` at the top level; without unflatten
        the AR recv used to see ``meta={}`` and leave prewarmed Talker
        ``resumable=True``.
        """
        if payload_data is None:
            return False, False
        if isinstance(payload_data, Mapping):
            nested = unflatten_payload(payload_data)
            meta = nested.get("meta", {})
            if isinstance(meta, Mapping):
                return (
                    cls._is_truthy_scalar(meta.get("finished")),
                    cls._is_truthy_scalar(meta.get("is_segment_finished")),
                )
            return (
                cls._is_truthy_scalar(getattr(meta, "finished", None)),
                cls._is_truthy_scalar(getattr(meta, "is_segment_finished", None)),
            )
        meta = getattr(payload_data, "meta", None)
        if meta is None:
            return False, False
        return (
            cls._is_truthy_scalar(getattr(meta, "finished", None)),
            cls._is_truthy_scalar(getattr(meta, "is_segment_finished", None)),
        )

    @staticmethod
    def _confirmed_num_computed_tokens(request: Request) -> int:
        # vLLM async scheduling advances num_computed_tokens with output
        # placeholders before the corresponding token is committed. Connector
        # chunk send watermarks must use only committed tokens.
        num_computed = int(getattr(request, "num_computed_tokens", 0))
        num_placeholders = int(getattr(request, "num_output_placeholders", 0) or 0)
        return max(0, num_computed - num_placeholders)

    @staticmethod
    def _refresh_generation_chunk_prefill_state(request: Request) -> None:
        request.num_prompt_tokens = len(request.prompt_token_ids)
        if getattr(request, "prefill_stats", None) is None:
            request.prefill_stats = PrefillStats()

    @classmethod
    def create_connector(cls, model_config: Any):
        connector_config = getattr(model_config, "stage_connector_config", None)
        if connector_config is None:
            connector_config = {}
        elif not isinstance(connector_config, dict):
            connector_config = {
                "name": getattr(connector_config, "name", None),
                "extra": getattr(connector_config, "extra", {}),
            }

        connector_specs = ConnectorSpec(
            name=connector_config.get("name", "SharedMemoryConnector"),
            extra=connector_config.get("extra", {}),
        )
        return OmniConnectorFactory.create_connector(connector_specs)

    def load_async(self, request: Request):
        """Register a request for asynchronous chunk retrieval.

        This method does not read from the connector directly. It records
        request metadata and enqueues the request id for the background
        receive loop to poll.

        Stage-0 has no upstream producer, so this call is a no-op there.

        Args:
            request: The request object needing data.
        """
        stage_id = self.connector.stage_id

        if stage_id == 0 or not self.receives_chunks:
            return
        if not hasattr(request, "additional_information"):
            request.additional_information = None
        self._cancelled_load_reqs.discard(request.request_id)
        self._pending_load_reqs.append(request)
        with self._recv_cond:
            self._recv_cond.notify()

    def save_async(
        self,
        multimodal_output: dict[str, Any] | None = None,
        request: Request | None = None,
        is_segment_finished: bool = False,
    ):
        """Build and enqueue one chunk for asynchronous sending.

        Payload extraction happens in ``_send_single_request`` on the
        background save_loop thread.

        For streaming input request ``is_segment_finished`` marks the end
        of the current realtime input segment. It is intentionally separate
        from ``request.is_finished()``: a resumable `/v1/realtime` session
        can finish one audio segment and later continue with another segment
        under the same external request id. For other requests, it is the same
        as ``request.is_finished()``.

        Args:
            multimodal_output: Per-request multimodal output dictionary
            request: Request object
            is_segment_finished: whether the segment of request is finished
        """
        is_finished = request.is_finished() and not request.resumable

        confirmed_num_computed_tokens = self._confirmed_num_computed_tokens(request)

        # If the request is preempted, skip the already saved chunks.
        if confirmed_num_computed_tokens < self.requests_num_chunks_sent.get(request.external_req_id, 0):
            logger.warning(
                f"Enqueue save_async for request {request.external_req_id}, "
                f"request.num_computed_tokens={request.num_computed_tokens}, "
                f"request.num_output_placeholders={getattr(request, 'num_output_placeholders', 0)}, "
                f"previous_chunks_sent={self.requests_num_chunks_sent.get(request.external_req_id, 0)}"
            )
            return

        self.requests_num_chunks_sent[request.external_req_id] = confirmed_num_computed_tokens
        task = {
            "multimodal_output": multimodal_output,
            "request": request,
            "is_finished": is_finished,
            "is_segment_finished": is_segment_finished,
        }
        stage_id = self.connector.stage_id
        self._pending_save_reqs.append(task)
        with self._save_cond:
            self._save_cond.notify()

    def _enqueue_pending_upstream_payload(
        self,
        req_id: str,
        payload_data: Any,
        *,
        finished: bool,
        segment_finished: bool,
        chunk_id: int,
    ) -> None:
        queued = dict(payload_data) if isinstance(payload_data, Mapping) else payload_data
        bucket = self._pending_upstream_payloads.setdefault(req_id, deque())
        bucket.append(
            {
                "payload": queued,
                "finished": finished,
                "segment_finished": segment_finished,
                "chunk_id": chunk_id,
            }
        )

    def try_apply_pending_upstream_payload(self, request: Request) -> bool:
        """Apply one queued Stage1 sentence after the live Talker segment stops."""
        req_id = request.request_id
        bucket = self._pending_upstream_payloads.get(req_id)
        if not bucket:
            return False
        # Mid-decode parking uses RUNNING / WAITING_FOR_CHUNK with computed still
        # past the prompt. Only WAITING (rearmed after Talker segment EOS) is a
        # safe boundary to open the next sentence as new_segment. Draining while
        # WAITING_FOR_CHUNK wiped TTS text / last-hidden and OOBd embeddings.
        if self._request_already_generating(request):
            if getattr(request, "status", None) != RequestStatus.WAITING:
                return False
        self.segment_finished_requests.discard(req_id)
        item = bucket.popleft()
        more_pending = bool(bucket)
        if not more_pending:
            self._pending_upstream_payloads.pop(req_id, None)
        payload_data = item["payload"]
        if isinstance(payload_data, Mapping):
            payload_data = unflatten_payload(payload_data)
        meta = _meta_to_dict(
            payload_data.get("meta") if isinstance(payload_data, Mapping) else None
        )
        # Only mark turn exhausted when this drained item is finished AND no
        # further queued sentences remain (defensive against out-of-order flags).
        apply_finished = bool(item.get("finished")) and not more_pending
        # Empty finish sentinels must not open a new Talker segment: they only
        # release the upstream wait gate and would wipe TTS text conditioning.
        # Return False so callers do NOT mark finished_load / ready_chunks —
        # that re-admits Talker as scheduled_new_reqs with stale computed
        # (Smoke3 CUDA indexSelect: prompt_len=34, computed=145).
        if not _payload_has_talker_conditioning(payload_data):
            if apply_finished:
                self.upstream_exhausted_requests.add(req_id)
                request.resumable = False
                # Post-EOS empty finish: do not finished_load / re-schedule.
                # Scheduler finishes via _terminal_empty_finish_reqs.
                self._terminal_empty_finish_reqs.add(req_id)
                # If Talker already left RUNNING (segment EOS → WAITING), do not
                # wait for the next schedule sweep — immediately tell Code2Wav
                # finished=True. Otherwise already_generating (stale computed)
                # can skip _finish_empty_prompt_chunk_requests and hang generate().
                if getattr(request, "status", None) != RequestStatus.RUNNING:
                    self._pending_save_reqs.append(
                        {
                            "multimodal_output": None,
                            "request": request,
                            "is_finished": True,
                            "is_segment_finished": True,
                        }
                    )
                    with self._save_cond:
                        self._save_cond.notify()
            if bool(item.get("segment_finished")):
                self.segment_finished_requests.add(req_id)
            logger.info(
                "[async_chunk] req=%s drained_pending_finish_sentinel finished=%s "
                "segment_finished=%s remaining_pending=%d (no talker conditioning; skip new_segment)",
                req_id,
                apply_finished,
                bool(item.get("segment_finished")),
                len(self._pending_upstream_payloads.get(req_id, ())),
            )
            return False
        applied = self._apply_ar_talker_payload(
            request,
            payload_data,
            finished=apply_finished,
            segment_finished=bool(item.get("segment_finished")),
            resolved_aura_payload=False,
            meta=meta,
            chunk_id=max(1, int(item.get("chunk_id") or 0)),
            new_segment=True,
        )
        logger.info(
            "[async_chunk] req=%s drained_pending_payload finished=%s segment_finished=%s "
            "remaining_pending=%d resumable=%s applied=%s",
            req_id,
            apply_finished,
            bool(item.get("segment_finished")),
            len(self._pending_upstream_payloads.get(req_id, ())),
            getattr(request, "resumable", False),
            applied,
        )
        return bool(applied)

    def _apply_ar_talker_payload(
        self,
        request: Request,
        payload_data: Any,
        *,
        finished: bool,
        segment_finished: bool,
        resolved_aura_payload: bool,
        meta: Mapping[str, Any],
        chunk_id: int,
        new_segment: bool = False,
    ) -> bool:
        """Apply Stage1→Talker payload. Returns True only when decode-ready work was applied."""
        req_id = request.request_id
        if not isinstance(payload_data, Mapping):
            return False
        # Belt-and-suspenders: never wipe live Talker decode state. Callers must
        # pass new_segment=True only after the previous segment has stopped.
        if not new_segment and not resolved_aura_payload and self._request_already_generating(request):
            self._enqueue_pending_upstream_payload(
                req_id,
                payload_data,
                finished=finished,
                segment_finished=segment_finished,
                chunk_id=chunk_id,
            )
            logger.warning(
                "[async_chunk] req=%s refuse_apply_mid_decode computed=%s prompt_len=%s "
                "finished=%s → queued pending=%d",
                req_id,
                getattr(request, "num_computed_tokens", None),
                len(getattr(request, "prompt_token_ids", []) or []),
                finished,
                len(self._pending_upstream_payloads.get(req_id, ())),
            )
            return False
        payload_data = dict(unflatten_payload(payload_data))
        merged_meta = _meta_to_dict(payload_data.get("meta"))
        for key, value in _meta_to_dict(meta).items():
            merged_meta.setdefault(key, value)
        meta = merged_meta
        if new_segment:
            # Refuse new_segment wipe when this payload cannot condition Talker.
            if not _payload_has_talker_conditioning(payload_data):
                if finished:
                    self.upstream_exhausted_requests.add(req_id)
                    request.resumable = False
                if segment_finished:
                    self.segment_finished_requests.add(req_id)
                logger.info(
                    "[async_chunk] req=%s skip_new_segment_empty_payload finished=%s "
                    "segment_finished=%s",
                    req_id,
                    finished,
                    segment_finished,
                )
                return False
            prompt_ids = payload_data.get("prompt_token_ids")
            if prompt_ids is None:
                ids = payload_data.get("ids", {})
                prompt_ids = ids.get("prompt") if isinstance(ids, Mapping) else None
            next_len = meta.get("next_stage_prompt_len")
            if not isinstance(next_len, int) or next_len <= 0:
                if isinstance(prompt_ids, list) and prompt_ids:
                    meta["next_stage_prompt_len"] = len(prompt_ids)
            # Only declare replace when we actually know the next prompt length.
            if isinstance(meta.get("next_stage_prompt_len"), int) and meta["next_stage_prompt_len"] > 0:
                meta["replace_streaming_prompt"] = True
            else:
                logger.warning(
                    "[async_chunk] req=%s new_segment missing next_stage_prompt_len; "
                    "skip replace_streaming_prompt to avoid wiping TTS state",
                    req_id,
                )
                if finished:
                    self.upstream_exhausted_requests.add(req_id)
                    request.resumable = False
                if segment_finished:
                    self.segment_finished_requests.add(req_id)
                return False
            payload_data["meta"] = meta

        if not new_segment:
            prompt_token_ids = payload_data.get("prompt_token_ids")
            if prompt_token_ids is None:
                prompt_token_ids = payload_data.get("ids", {}).get("prompt") if isinstance(payload_data.get("ids"), Mapping) else None
            if isinstance(prompt_token_ids, list) and all(isinstance(token_id, int) for token_id in prompt_token_ids):
                _replace_request_prompt_token_ids(request, prompt_token_ids)

        prev_info = getattr(request, "additional_information", None)
        if isinstance(prev_info, dict):
            prev_tts_info = {key: value for key, value in prev_info.items() if str(key).startswith("tts_")}
            payload_additional_info = payload_data.get("additional_information")
            if prev_tts_info and isinstance(payload_additional_info, dict):
                payload_data = dict(payload_data)
                payload_data["additional_information"] = {**prev_tts_info, **payload_additional_info}
        request.omni_stage_payload = payload_data
        if resolved_aura_payload:
            slim_info = payload_data.get("additional_information")
            request.additional_information = slim_info if isinstance(slim_info, dict) else {}
        else:
            request.additional_information = payload_data

        replace_prompt = meta.get("replace_streaming_prompt") is True
        if new_segment or (getattr(request, "resumable", False) and (chunk_id > 0 or replace_prompt)):
            construct_next_stage_streaming_input_prompt(payload_data, request)

        if not resolved_aura_payload:
            _apply_max_new_tokens_from_payload(request, payload_data)

        if finished:
            self.upstream_exhausted_requests.add(req_id)
            request.resumable = False
        if segment_finished:
            self.segment_finished_requests.add(req_id)
        if finished or getattr(request, "resumable", False):
            logger.info(
                "[async_chunk] recv req=%s finished=%s segment_finished=%s resumable=%s new_segment=%s",
                req_id,
                finished,
                segment_finished,
                getattr(request, "resumable", False),
                new_segment,
            )
        if not resolved_aura_payload:
            _mark_async_chunk_stage_ready(request)
        return True

    def _poll_single_request(self, request: Request):
        stage_id = self.connector.stage_id
        target_stage_id = stage_id - 1
        req_id = request.request_id
        if self.model_mode == "ar" and self.try_apply_pending_upstream_payload(request):
            self._finished_load_reqs.add(req_id)
            return True
        chunk_id = self.get_req_chunk[req_id]
        external_req_id = self.request_ids_mapping.get(req_id, req_id)
        connector_get_key = f"{external_req_id}_{target_stage_id}_{chunk_id}"

        # Use timeout=0 for non-blocking poll
        try:
            result = self.connector.get(
                str(target_stage_id),
                str(stage_id),
                connector_get_key,
            )
        except Exception as e:
            logger.error(f"SharedMemoryConnector get failed for req {connector_get_key}: {e}")
            return False

        if result is None:
            return False
        payload_data, size = result

        if payload_data:
            # Update connector state
            self.get_req_chunk[req_id] += 1

            if isinstance(payload_data, Mapping):
                payload_data = unflatten_payload(payload_data)
            payload_finished, payload_segment_finished = self._payload_terminal_flags(payload_data)
            meta = payload_data.get("meta", {}) if isinstance(payload_data, Mapping) else {}
            if not isinstance(meta, Mapping):
                meta = {}
            resolved_aura_payload = False
            if self.model_mode == "ar" and "aura_asr_transcript" in payload_data:
                from vllm_omni.model_executor.stage_input_processors.aura_omni import (
                    resolve_aura_async_chunk_stage_payload,
                )

                # Stamp before resolve so Stage-1 local TTFT includes
                # process_inputs / mm expand (sync-aligned).
                _mark_async_chunk_stage_ready(request)
                resolve_aura_async_chunk_stage_payload(
                    payload_data,
                    request,
                    self.model_config,
                    vllm_config=self.vllm_config,
                )
                resolved_aura_payload = True
                mm_features = getattr(request, "mm_features", None)
                if self.mm_receiver_cache is not None and mm_features:
                    request.mm_features = self.mm_receiver_cache.get_and_update_features(
                        list(mm_features)
                    )
            if self.model_mode == "ar":
                # Any late Stage1 sentence that arrives while Talker has already
                # produced tokens must not be applied as a same-segment update:
                # that replaces additional_information (wiping hidden_states['last'])
                # and clamps max_new_tokens mid-flight (Smoke3 CUDA / RuntimeError).
                # Queue it. Drain immediately only on true between-segment WAITING
                # (Talker EOS rearm). WAITING_FOR_CHUNK is mid-decode parking and
                # must keep the live TTS payload until the segment actually stops.
                already_generating = self._request_already_generating(request)
                if already_generating and not resolved_aura_payload:
                    self._enqueue_pending_upstream_payload(
                        req_id,
                        payload_data,
                        finished=payload_finished,
                        segment_finished=payload_segment_finished,
                        chunk_id=chunk_id,
                    )
                    logger.info(
                        "[async_chunk] recv req=%s queue_late_payload finished=%s "
                        "segment_finished=%s status=%s computed=%s prompt_len=%s pending=%d",
                        req_id,
                        payload_finished,
                        payload_segment_finished,
                        getattr(request, "status", None),
                        getattr(request, "num_computed_tokens", None),
                        len(getattr(request, "prompt_token_ids", []) or []),
                        len(self._pending_upstream_payloads.get(req_id, ())),
                    )
                    if getattr(request, "status", None) == RequestStatus.WAITING:
                        if self.try_apply_pending_upstream_payload(request):
                            self._finished_load_reqs.add(req_id)
                            return True
                    if getattr(request, "status", None) == RequestStatus.WAITING_FOR_CHUNK:
                        # Mid-decode WFC + queued sentence: do NOT mark finished_load.
                        # finished_load → ready_chunks re-admits Talker as
                        # scheduled_new_reqs with stale prompt/KV lengths
                        # (Smoke3 CUDA indexSelect). process_chunk_queue resumes
                        # SAME-segment RUNNING without advertising a new chunk.
                        return False
                    # RUNNING: stay on the running queue and keep polling.
                    return False
                # After segment EOS, computed may still sit past prompt while
                # outputs are cleared. Open the next sentence as new_segment so
                # replace_streaming_prompt resets computed before max_new_tokens.
                prompt_len = int(getattr(request, "num_prompt_tokens", 0) or 0)
                if prompt_len <= 0:
                    prompt_len = len(getattr(request, "prompt_token_ids", None) or [])
                computed = int(getattr(request, "num_computed_tokens", 0) or 0)
                open_new_segment = computed > prompt_len > 0
                applied = self._apply_ar_talker_payload(
                    request,
                    payload_data,
                    finished=payload_finished,
                    segment_finished=payload_segment_finished,
                    resolved_aura_payload=resolved_aura_payload,
                    meta=meta,
                    chunk_id=chunk_id,
                    new_segment=open_new_segment,
                )
                if not applied:
                    # Empty finish / refuse paths must not advertise ready_chunks.
                    # Mark terminal so the AR scheduler finishes the WAITING
                    # post-EOS request instead of hanging for another chunk.
                    if payload_finished or req_id in self.upstream_exhausted_requests:
                        self.upstream_exhausted_requests.add(req_id)
                        request.resumable = False
                        self._terminal_empty_finish_reqs.add(req_id)
                        logger.info(
                            "[async_chunk] req=%s terminal_empty_finish "
                            "computed=%s prompt_len=%s (scheduler will finish)",
                            req_id,
                            computed,
                            prompt_len,
                        )
                    return False
            else:
                if payload_finished:
                    self.upstream_exhausted_requests.add(req_id)
                    request.resumable = False
                if payload_segment_finished:
                    self.segment_finished_requests.add(req_id)
                if payload_finished or payload_segment_finished:
                    logger.info(
                        "[async_chunk] recv stage=%s req=%s finished=%s segment_finished=%s resumable=%s",
                        stage_id,
                        req_id,
                        payload_finished,
                        payload_segment_finished,
                        getattr(request, "resumable", False),
                    )

                new_ids = payload_data.get("codes", {}).get("audio")
                has_tensor_codes = isinstance(new_ids, torch.Tensor)
                use_tensor_codes = has_tensor_codes and new_ids.ndim >= 2
                prompt_token_ids: list[int]
                if use_tensor_codes:
                    prompt_token_ids = [0] if new_ids.numel() > 0 else []
                elif has_tensor_codes:
                    new_ids = new_ids.tolist()
                    prompt_token_ids = new_ids
                elif new_ids is None:
                    new_ids = []
                    prompt_token_ids = new_ids
                else:
                    prompt_token_ids = new_ids
                request.prompt_token_ids = prompt_token_ids
                prev_info = getattr(request, "additional_information", None)
                info = dict(prev_info) if isinstance(prev_info, dict) else {}
                for key, value in payload_data.items():
                    if key == "codes":
                        if use_tensor_codes and isinstance(value, dict):
                            existing_sub = info.get(key)
                            merged_sub = dict(existing_sub) if isinstance(existing_sub, dict) else {}
                            merged_sub.update(value)
                            info[key] = merged_sub
                        continue
                    if isinstance(value, dict):
                        existing_sub = info.get(key)
                        merged_sub = dict(existing_sub) if isinstance(existing_sub, dict) else {}
                        # Keep the latest meta.finished. Skipping it left Code2Wav
                        # additional_information stuck at False, so model outputs
                        # never finished and generate() hung after spoken leftover.
                        merged_sub.update(value)
                        info[key] = merged_sub
                        continue
                    info[key] = value
                request.additional_information = info
                request.num_computed_tokens = 0

                # Empty chunk with more data expected: keep polling.
                has_new_ids = bool(new_ids.numel()) if use_tensor_codes else bool(new_ids)
                if not has_new_ids and payload_segment_finished:
                    # Preserve an explicit scheduler boundary even when it
                    # contains no new codec frames.
                    request.prompt_token_ids = [0]
                if not has_new_ids and not payload_finished and not payload_segment_finished:
                    # The base recv loop treats False as "not ready yet" and
                    # requeues the request. Do not mark an empty non-terminal
                    # chunk as ready, otherwise Stage1 can consume before the
                    # first DAC frame arrives.
                    return False
                self._refresh_generation_chunk_prefill_state(request)
                _mark_async_chunk_stage_ready(request)

            # Mark as finished for consumption
            self._finished_load_reqs.add(req_id)
            return True

        return False

    def _send_single_request(self, task: dict):
        raw_mm = task["multimodal_output"]
        multimodal_output = unflatten_payload(raw_mm) if isinstance(raw_mm, Mapping) else raw_mm
        request = task["request"]
        is_finished = task["is_finished"]
        is_segment_finished = task["is_segment_finished"]
        stage_id = self.connector.stage_id
        next_stage_id = stage_id + 1
        external_req_id = request.external_req_id
        chunk_id = self.put_req_chunk[external_req_id]
        connector_put_key = f"{external_req_id}_{stage_id}_{chunk_id}"
        # Process payload in save_loop thread
        payload_data: OmniPayloadStruct | dict[str, Any] | None = None
        processor_name = getattr(self.custom_process_next_stage_input_func, "__name__", None)
        if self.custom_process_next_stage_input_func:
            try:
                payload_data = self.custom_process_next_stage_input_func(
                    transfer_manager=self,
                    multimodal_output=multimodal_output,
                    request=request,
                    # Existing processors use is_finished as a flush signal.
                    # Terminal stops no longer count as segment boundaries
                    # (is_segment_finished is False when the request finishes,
                    # see #5383), but the processor must still flush its
                    # accumulated tail on the terminal chunk — otherwise the
                    # downstream stage receives the finished marker without
                    # the final payload (#5413).
                    is_finished=is_segment_finished or is_finished,
                )

            except Exception as e:
                logger.error(
                    "Chunk transfer processor failed at stage=%s processor=%s ext_req=%s: %s",
                    stage_id,
                    processor_name,
                    external_req_id,
                    e,
                )

        if payload_data is None:
            if not (is_segment_finished or is_finished):
                return
            # Segment/request finish markers must still reach downstream even when
            # the processor has no tensor payload.
            payload_data = OmniPayloadStruct()
        # Mid-gen Sentence TTS: processor returned a real Talker prompt while Stage1
        # is still decoding. Downstream Talker must see a complete segment (with
        # text conditioning); otherwise decode crashes with missing text.
        force_segment_finished = False
        if (
            not is_segment_finished
            and not is_finished
            and processor_name == "aura2tts_async_chunk"
            and payload_data is not None
        ):
            is_segment_finished = True
            force_segment_finished = True
        if isinstance(payload_data, dict):
            meta = payload_data.setdefault("meta", {})
            if not isinstance(meta, dict):
                meta = {}
                payload_data["meta"] = meta
            # Preserve processor-set terminal marker. Prewarmed Talker/Code2Wav
            # stay resumable=True until meta.finished arrives; save_async then
            # computes is_finished=request.is_finished() and not resumable, which
            # is False for a resumable Stage1 segment stop and would otherwise
            # stomp aura2tts meta.finished=True. Mid-gen sentence TTS leaves
            # processor finished=False, so this OR stays False.
            processor_finished = self._is_truthy_scalar(meta.get("finished"))
            meta["finished"] = torch.tensor(bool(is_finished or processor_finished), dtype=torch.bool)
            if processor_name == "aura2tts_async_chunk" or processor_finished or is_finished:
                logger.info(
                    "[async_chunk] send stage=%s proc=%s ext=%s task_finished=%s "
                    "processor_finished=%s wire_finished=%s segment_finished=%s",
                    stage_id,
                    processor_name,
                    external_req_id,
                    is_finished,
                    processor_finished,
                    bool(is_finished or processor_finished),
                    is_segment_finished,
                )
            # Respect processor-set segment boundary (#5383) unless AURA mid-gen
            # Sentence TTS forced a complete Talker segment above.
            if force_segment_finished or meta.get("is_segment_finished") is None:
                meta["is_segment_finished"] = torch.tensor(is_segment_finished, dtype=torch.bool)
        else:
            if payload_data.meta is None:
                payload_data.meta = MetaStruct()
            # talker2code2wav emits leftover tails with MetaStruct.finished=True
            # while save_async may still snapshot is_finished=False. Blind
            # overwrite left Code2Wav waiting forever after spoken TTS (live:
            # 4 chunks, last leftover frames=131, then ~180s hang). Keep
            # processor finished only when Talker is no longer resumable —
            # mid-gen sentence leftover stays wire False.
            processor_finished = self._is_truthy_scalar(getattr(payload_data.meta, "finished", None))
            request_resumable = bool(getattr(request, "resumable", False))
            wire_finished = bool(is_finished or (processor_finished and not request_resumable))
            payload_data.meta.finished = torch.tensor(wire_finished, dtype=torch.bool)
            if processor_name == "talker2code2wav_async_chunk" or processor_finished or is_finished:
                logger.info(
                    "[async_chunk] send stage=%s proc=%s ext=%s task_finished=%s "
                    "processor_finished=%s wire_finished=%s resumable=%s segment_finished=%s",
                    stage_id,
                    processor_name,
                    external_req_id,
                    is_finished,
                    processor_finished,
                    wire_finished,
                    request_resumable,
                    is_segment_finished,
                )
            if force_segment_finished or payload_data.meta.is_segment_finished is None:
                payload_data.meta.is_segment_finished = torch.tensor(
                    is_segment_finished, dtype=torch.bool
                )

        success, size, metadata = self.connector.put(
            from_stage=str(stage_id),
            to_stage=str(next_stage_id),
            put_key=connector_put_key,
            data=payload_data,
        )

        if success:
            self.put_req_chunk[external_req_id] += 1
            self.ramp_chunk_count[external_req_id] += 1
            logger.debug(f"[Stage-{stage_id}] Sent {connector_put_key}")
            # Sender uses struct attr access here; the receive path in
            # `_load_one_request` / `_update_request_payload` reads dict keys.
            # That asymmetry is intentional: `OmniMsgpackDecoder` is type-erased
            # (no target type), so the wire round-trips struct -> dict. If you
            # change the schema, update both ends — see test_wire_round_trip.
            if isinstance(payload_data, dict):
                meta = payload_data.get("meta", {})
                finished_flag = meta.get("finished") if isinstance(meta, dict) else None
            else:
                finished_flag = payload_data.meta.finished if payload_data.meta is not None else None
            is_payload_finished = False
            if isinstance(finished_flag, torch.Tensor):
                is_payload_finished = finished_flag.numel() == 1 and bool(finished_flag.item())
            elif finished_flag is not None:
                is_payload_finished = bool(finished_flag)

            # Reclaim per-request async state only after the terminal payload
            # has been sent successfully. This avoids cleanup->save races.
            if is_payload_finished:
                self.cleanup(request.request_id, external_req_id)

        if is_segment_finished:
            self.code_prompt_token_ids.pop(external_req_id, None)
            self.requests_num_chunks_sent.pop(external_req_id, None)
            self.ramp_chunk_count.pop(external_req_id, None)
            cached_ic = getattr(self, "_cached_ic", None)
            if cached_ic is not None:
                cached_ic.pop(external_req_id, None)

    @staticmethod
    def _request_already_generating(request: Any) -> bool:
        """True only while Talker is mid-segment decode.

        ``computed > prompt`` alone is not enough: after segment EOS the
        scheduler clears ``_output_token_ids`` but leaves ``num_computed_tokens``
        past the prompt. Treating that as mid-decode queues the next Stage1
        sentence under ``WAITING_FOR_CHUNK`` and never drains it (Smoke3 hang).
        """
        prompt_len = int(getattr(request, "num_prompt_tokens", 0) or 0)
        if prompt_len <= 0:
            prompt_len = len(getattr(request, "prompt_token_ids", None) or [])
        computed = int(getattr(request, "num_computed_tokens", 0) or 0)
        if not (computed > prompt_len > 0):
            return False
        output_ids = getattr(request, "_output_token_ids", None)
        if output_ids is not None and len(output_ids) == 0:
            return False
        return True

    def is_done_receiving_chunks(self, request_id: str) -> bool:
        """Return True if the request should stop polling upstream chunks.

        Covers both the whole-request marker (``upstream_exhausted_requests``)
        and the per-segment marker (``segment_finished_requests``) used while
        waiting for the next streaming input slice. Neither means this
        stage's own generation is done -- see vllm-project/vllm-omni#5349.
        """
        return request_id in self.upstream_exhausted_requests or request_id in self.segment_finished_requests

    ########################################################################
    # Cleanup
    ########################################################################

    def cleanup_receiver(self, request_id: str) -> None:
        """Reclaim receiver-side per-request state (keyed by internal id).

        Safe to call from the scheduler even when ``save_async()`` has
        enqueued work that the background thread has not yet processed,
        because it only touches receiver-side dictionaries.

        Must also purge the request from the chunk-parking deques
        (``waiting_for_chunk_waiting_requests`` / ``_running_requests`` /
        ``_held_non_active``): otherwise a caller that calls
        ``restore_queues()`` without ``scheduler_requests`` (e.g. a unit
        test, or any future caller not synced with the scheduler's own
        request-removal timing) would re-admit an already-finished
        request into the visible queue, which ``_promote_active_streams``
        would then FIFO-promote ahead of genuinely-waiting requests. See
        vllm-project/vllm-omni#5349's active-stream-window tests.

        Idempotent: calling with an already-cleaned or unknown id is safe.
        """
        self._active_streams.pop(request_id, None)
        self.upstream_exhausted_requests.discard(request_id)
        self._terminal_empty_finish_reqs.discard(request_id)
        self._pending_upstream_payloads.pop(request_id, None)
        self.segment_finished_requests.discard(request_id)
        self.get_req_chunk.pop(request_id, None)
        self.requests_with_ready_chunks.discard(request_id)
        self.request_ids_mapping.pop(request_id, None)
        self.requests_origin_status.pop(request_id, None)
        self._discard_from_chunk_deque(self.waiting_for_chunk_waiting_requests, request_id)
        self._discard_from_chunk_deque(self.waiting_for_chunk_running_requests, request_id)
        self._discard_from_chunk_deque(self._held_non_active, request_id)

        self._cancelled_load_reqs.add(request_id)
        self._finished_load_reqs.discard(request_id)

    @staticmethod
    def _discard_from_chunk_deque(deque_list: deque[Any], request_id: str) -> None:
        if not deque_list:
            return
        for _ in range(len(deque_list)):
            request = deque_list.popleft()
            if request.request_id != request_id:
                deque_list.append(request)

    def cleanup_sender(self, external_req_id: str) -> None:
        """Reclaim sender-side per-request state (keyed by external id).

        Must only be called after the terminal chunk has actually been
        sent (i.e. from ``_send_single_request``), not before.

        Idempotent: calling with an already-cleaned or unknown id is safe.
        """
        self.put_req_chunk.pop(external_req_id, None)
        self.request_payload.pop(external_req_id, None)
        self.code_prompt_token_ids.pop(external_req_id, None)
        self.requests_num_chunks_sent.pop(external_req_id, None)
        self.ramp_chunk_count.pop(external_req_id, None)
        self._pending_streaming_prefills.pop(external_req_id, None)

        cached_ic = getattr(self, "_cached_ic", None)
        if cached_ic is not None:
            cached_ic.pop(external_req_id, None)

    def cleanup(
        self,
        request_id: str,
        external_req_id: str | None = None,
    ) -> None:
        """Reclaim all per-request state after a request finishes.

        Idempotent: calling with an already-cleaned or unknown id is safe.

        Args:
            request_id: Internal request id (receive / scheduler side key).
            external_req_id: External request id (send / payload side key).
                When *None*, looked up from ``request_ids_mapping``.
        """
        if external_req_id is None:
            external_req_id = self.request_ids_mapping.get(request_id, request_id)

        self.cleanup_receiver(request_id)
        self.cleanup_sender(external_req_id)

    ########################################################################
    # Schedule Helper
    ########################################################################

    def process_pending_chunks(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
        *,
        scheduler_requests: dict[str, Request] | None = None,
    ) -> None:
        """
        Process pending chunks for waiting and running queues.

        When ``scheduler_requests`` is provided, purges any
        ``waiting_for_chunk_*_requests`` deque entries whose
        ``request_id`` is no longer tracked by it (e.g. after a
        mid-flight abort that ran ``Scheduler._free_request``) before
        processing chunks. Without this purge, ``restore_queues`` would
        later re-inject the freed ``Request`` onto ``running_queue`` and
        the worker's ``_update_states`` would crash with ``KeyError``
        reading ``self.requests[req_id]``. See vllm-project/vllm-omni#3736.

        ``scheduler_requests`` is keyword-only and optional; production
        schedulers always pass their live request map, while legacy
        callers that don't track aborts may omit it to keep the prior
        (unguarded) behaviour.
        """
        if not self.receives_chunks:
            return
        if self.connector.stage_id == 0:
            return

        # Purge deque entries whose request was freed mid-flight (abort →
        # Scheduler._free_request) before any chunk processing, so neither
        # the legacy nor the active-stream path can re-inject a zombie
        # Request onto the queues. See vllm-project/vllm-omni#3736.
        if scheduler_requests is not None:
            self._purge_untracked_chunk_requests(self.waiting_for_chunk_waiting_requests, scheduler_requests)
            self._purge_untracked_chunk_requests(self.waiting_for_chunk_running_requests, scheduler_requests)

        if self._active_window <= 0:
            self._process_chunk_queue_legacy(
                waiting_queue, self.waiting_for_chunk_waiting_requests, RequestStatus.WAITING, self._finished_load_reqs
            )
            self._process_chunk_queue_legacy(
                running_queue,
                self.waiting_for_chunk_running_requests,
                RequestStatus.RUNNING,
                self._finished_load_reqs,
            )
            while len(running_queue) > self.scheduler_max_num_seqs:
                request = running_queue.pop()
                request.status = RequestStatus.PREEMPTED
                waiting_queue.prepend_requests([request])
            return

        self._promote_active_streams(running_queue)
        self._promote_active_streams(waiting_queue)
        self._process_chunk_queue(
            waiting_queue, self.waiting_for_chunk_waiting_requests, RequestStatus.WAITING, self._finished_load_reqs
        )
        self._process_chunk_queue(
            running_queue, self.waiting_for_chunk_running_requests, RequestStatus.RUNNING, self._finished_load_reqs
        )
        self._promote_active_streams(waiting_queue)
        self._preempt_non_active_running(waiting_queue, running_queue)

    def _promote_active_streams(self, queue: Any) -> None:
        if len(self._active_streams) >= self._active_window:
            return
        for request in list(queue):
            if len(self._active_streams) >= self._active_window:
                return
            request_id = request.request_id
            if request_id in self._active_streams:
                continue
            # Iterating the existing queue preserves FIFO admission.
            self._active_streams[request_id] = request

    def _ensure_active_stream(self, request: Request) -> bool:
        if self._active_window <= 0:
            return True
        request_id = request.request_id
        if request_id in self._active_streams:
            self._active_streams[request_id] = request
            return True
        if len(self._active_streams) >= self._active_window:
            return False
        self._active_streams[request_id] = request
        return True

    @property
    def num_running_waiting_for_chunk(self) -> int:
        """Count running requests temporarily removed while awaiting a chunk."""
        return len(self.waiting_for_chunk_running_requests)

    def _preempt_non_active_running(self, waiting_queue: Any, running_queue: list[Request]) -> None:
        # Hold non-active running requests in a private deque rather than
        # routing them back through waiting_queue. Routing through the
        # vllm RequestQueue mid-step triggers
        #   "Cannot register new removed request after self.removed has
        #    been read"
        # in vllm.v1.sample.logits_processor.state when the persistent
        # batch was already snapshotted. They are returned to
        # running_queue in restore_queues() so the next scheduler tick
        # re-evaluates them through _promote_active_streams.
        index = len(running_queue) - 1
        while index >= 0:
            request = running_queue[index]
            if request.request_id in self._active_streams:
                index -= 1
                continue
            request = running_queue.pop(index)
            self._held_non_active.append(request)
            index -= 1

    def _process_chunk_queue_legacy(
        self,
        queue: Any,
        waiting_for_chunk_list: deque[Any],
        target_status: RequestStatus,
        finished_load_reqs: set[str],
    ) -> None:
        queue_snapshot = list(queue)
        for request in queue_snapshot:
            if request.status != RequestStatus.WAITING_FOR_CHUNK:
                if request.request_id in self.requests_with_ready_chunks:
                    # Requests that have loaded chunk from last round
                    # of schedule, but have not scheduled
                    continue
                if self.is_done_receiving_chunks(request.request_id):
                    # segment_finished is set when Stage1 emits a sentence; Talker
                    # may still be decoding that segment. Do not wipe TTS state
                    # mid-decode (would drop text conditioning / last-hidden).
                    if self._request_already_generating(request):
                        if target_status == RequestStatus.RUNNING:
                            self.load_async(request)
                        continue
                    request.additional_information = None
                    continue
                if target_status == RequestStatus.RUNNING and self._request_already_generating(request):
                    self.load_async(request)
                    continue
                # Requests that waiting for chunk
                self.load_async(request)
                request.status = RequestStatus.WAITING_FOR_CHUNK
            else:
                if request.request_id in finished_load_reqs:
                    request.status = target_status
                    finished_load_reqs.remove(request.request_id)
                    self.requests_with_ready_chunks.add(request.request_id)
                    continue
                # Mid-decode WFC: resume SAME segment without ready_chunks.
                # finished_load/ready_chunks would re-admit as scheduled_new_reqs
                # with stale computed (Smoke3 CUDA indexSelect).
                if self._request_already_generating(request):
                    request.status = target_status
                    continue
            queue.remove(request)
            self.requests_origin_status[request.request_id] = target_status
            waiting_for_chunk_list.append(request)

    def _purge_untracked_chunk_requests(
        self,
        deque_list: deque[Any],
        scheduler_requests: dict[str, Request],
    ) -> None:
        """Drop deque entries whose ``request_id`` is not in
        ``scheduler_requests`` and reclaim their receiver-side state.

        Handles requests that were aborted mid-flight while parked in a
        chunk-transfer deque: ``Scheduler._free_request`` deleted the
        entry from ``scheduler.requests`` but the deque still holds a
        reference to the now-freed ``Request``. Order of survivors is
        preserved.
        """
        if not deque_list:
            return
        for _ in range(len(deque_list)):
            request = deque_list.popleft()
            if request.request_id in scheduler_requests:
                deque_list.append(request)
            else:
                self.cleanup_receiver(request.request_id)

    def restore_queues(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
        scheduler_requests: dict[str, Request] | None = None,
    ) -> None:
        """
        Restore requests waiting for chunk to the waiting and running queues.

        Re-runs the zombie purge first to close the race window where an
        abort fires *between* ``process_pending_chunks`` and the
        ``finally``-clause ``restore_queues`` call. Without the second
        purge, ``running_queue.extend(...)`` would still re-inject a
        freed ``Request`` and crash the worker on the next tick.

        ``scheduler_requests`` is optional for back-compat with legacy
        callers (older tests pass only the two queue arguments). When
        provided, it gates both the deque purge and the per-request
        admit checks below; when ``None``, the purge is skipped and
        every parked request is restored unconditionally (the
        pre-purge behavior).
        """
        if not self.receives_chunks:
            return
        if scheduler_requests is not None:
            self._purge_untracked_chunk_requests(self.waiting_for_chunk_waiting_requests, scheduler_requests)
            self._purge_untracked_chunk_requests(self.waiting_for_chunk_running_requests, scheduler_requests)
        # Add request waiting for chunk to the waiting and running queue
        for request in self.waiting_for_chunk_waiting_requests:
            if scheduler_requests is None or request.request_id in scheduler_requests:
                waiting_queue.add_request(request)
        self.waiting_for_chunk_waiting_requests = deque()

        if self.waiting_for_chunk_running_requests:
            live_running_requests = [
                request
                for request in self.waiting_for_chunk_running_requests
                if scheduler_requests is None or request.request_id in scheduler_requests
            ]
            running_queue.extend(live_running_requests)
        self.waiting_for_chunk_running_requests = deque()

        if self._held_non_active:
            running_queue.extend(self._held_non_active)
            self._held_non_active = deque()

    def postprocess_scheduler_output(
        self,
        scheduler_output: Any,
        requests: dict[str, Request] | None = None,
    ) -> None:
        """
        Add additional info for cached requests and
        clean up ready chunks from scheduler output.
        """
        if not self.receives_chunks:
            return
        stage_id = self.connector.stage_id

        if stage_id == 0:
            return

        if requests is not None:
            self.attach_cached_additional_information(scheduler_output, requests)
        self._clear_chunk_ready(scheduler_output)

    @staticmethod
    def attach_cached_additional_information(scheduler_output: Any, requests: dict[str, Request]) -> None:
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if not cached_reqs:
            return
        if not hasattr(cached_reqs, "additional_information"):
            cached_reqs.additional_information = {}
        for req_id in cached_reqs.req_ids:
            request = requests.get(req_id) if req_id else None
            additional_info = getattr(request, "additional_information", None) if request else None
            cached_reqs.additional_information[req_id] = additional_info
            if request and additional_info:
                request.additional_information = None

    def _process_chunk_queue(
        self,
        queue: Any,
        waiting_for_chunk_list: deque[Any],
        target_status: RequestStatus,
        finished_load_reqs: set[str],
    ) -> None:
        queue_snapshot = list(queue)
        for request in queue_snapshot:
            if not self._ensure_active_stream(request):
                if target_status == RequestStatus.WAITING:
                    # A non-active placeholder must not remain visible to the
                    # scheduler: it has no connector payload yet, so running
                    # it would execute the downstream model with empty
                    # additional_information. Park it until restore_queues()
                    # and retry admission on the next scheduler tick.
                    queue.remove(request)
                    waiting_for_chunk_list.append(request)
                continue
            if request.status != RequestStatus.WAITING_FOR_CHUNK:
                if request.request_id in self.requests_with_ready_chunks:
                    # Requests that have loaded chunk from last round
                    # of schedule, but have not scheduled
                    continue
                if self.is_done_receiving_chunks(request.request_id):
                    # segment_finished is set when Stage1 emits a sentence; Talker
                    # may still be decoding that segment. Do not wipe TTS state
                    # mid-decode (would drop text conditioning / last-hidden).
                    if self._request_already_generating(request):
                        if target_status == RequestStatus.RUNNING:
                            self.load_async(request)
                        continue
                    request.additional_information = None
                    continue
                if target_status == RequestStatus.RUNNING and self._request_already_generating(request):
                    # Stay on the running queue; still poll so the next
                    # Stage1 sentence can be queued without rescheduling.
                    self.load_async(request)
                    continue
                # Requests that waiting for chunk
                self.load_async(request)
                request.status = RequestStatus.WAITING_FOR_CHUNK
            else:
                if request.request_id in finished_load_reqs:
                    request.status = target_status
                    finished_load_reqs.remove(request.request_id)
                    self.requests_with_ready_chunks.add(request.request_id)
                    continue
                # Mid-decode WFC: resume SAME segment without ready_chunks.
                # Advertising ready_chunks re-prefills as scheduled_new_reqs
                # with stale computed (Smoke3 CUDA indexSelect).
                if self._request_already_generating(request):
                    request.status = target_status
                    continue
            queue.remove(request)
            self.requests_origin_status[request.request_id] = target_status
            waiting_for_chunk_list.append(request)

    def _clear_chunk_ready(self, scheduler_output: Any) -> None:
        if scheduler_output.scheduled_new_reqs:
            for req_data in scheduler_output.scheduled_new_reqs:
                if req_data.req_id in self.requests_with_ready_chunks:
                    self.requests_with_ready_chunks.remove(req_data.req_id)

        if scheduler_output.scheduled_cached_reqs:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                if req_id in self.requests_with_ready_chunks:
                    self.requests_with_ready_chunks.remove(req_id)

    def finish_requests(
        self, request_ids: Any, finished_status: RequestStatus, requests: dict[str, Request] | None = None
    ) -> list[tuple[str, int]]:
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = requests.keys()

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = requests.get(req_id) if requests else None
            if request is None or request.is_finished():
                # Invalid request ID.
                continue
            if req_id in self.requests_origin_status:
                request.status = self.requests_origin_status.pop(req_id)

        request_ids = set(request_ids)

        self.waiting_for_chunk_waiting_requests = deque(
            request for request in self.waiting_for_chunk_waiting_requests if request.request_id not in request_ids
        )
        self.waiting_for_chunk_running_requests = deque(
            request for request in self.waiting_for_chunk_running_requests if request.request_id not in request_ids
        )
        self._held_non_active = deque(
            request for request in self._held_non_active if request.request_id not in request_ids
        )

        for req_id in request_ids:
            self._active_streams.pop(req_id, None)
            self.requests_with_ready_chunks.discard(req_id)
            self.upstream_exhausted_requests.discard(req_id)
            self._terminal_empty_finish_reqs.discard(req_id)
            self._finished_load_reqs.discard(req_id)
            self._cancelled_load_reqs.add(req_id)

        return []
