import contextlib
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

import vllm_omni.worker.gpu_generation_model_runner as gen_runner_module
from vllm_omni.worker.gpu_generation_model_runner import (
    ExecuteModelState,
    GPUGenerationModelRunner,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyInputBatch:
    def __init__(self, num_reqs=1):
        self.req_ids = [f"req-{i}" for i in range(num_reqs)]
        self.req_id_to_index = {req_id: i for i, req_id in enumerate(self.req_ids)}
        self.num_reqs = num_reqs
        self.vocab_size = 10


def _make_runner(multimodal_outputs, num_reqs=1):
    runner = object.__new__(GPUGenerationModelRunner)
    runner.execute_model_state = ExecuteModelState(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        multimodal_outputs,
        None,
    )
    runner.kv_connector_output = None
    runner.input_batch = _DummyInputBatch(num_reqs)
    runner.use_async_scheduling = False
    runner.device = torch.device("cpu")
    runner.supports_mm_inputs = False
    runner.speculative_config = None
    runner.routed_experts_initialized = False
    runner._async_chunk = False
    return runner


def test_sample_tokens_tensor_output():
    multimodal_outputs = torch.randn(1, 2, 3)
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.multimodal_outputs) == 1
    assert output.multimodal_outputs[0]["model_outputs"].shape == (2, 3)


def test_sample_tokens_list_output():
    multimodal_outputs = [torch.randn(2, 1)]
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.multimodal_outputs) == 1
    assert output.multimodal_outputs[0]["model_outputs"].shape == (2, 1)


def test_sample_tokens_list_allows_none_output():
    multimodal_outputs = [None]
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.multimodal_outputs) == 1
    assert output.multimodal_outputs[0]["model_outputs"] is None


def test_sample_tokens_dict_output():
    multimodal_outputs = {"audio": torch.randn(1, 4), "unused": None}
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.multimodal_outputs) == 1
    assert "audio" in output.multimodal_outputs[0]
    assert "unused" not in output.multimodal_outputs[0]
    assert output.multimodal_outputs[0]["audio"].shape == (1, 4)


def test_sample_tokens_broadcasts_single_empty_sentinel_to_batch():
    empty = torch.zeros(0)
    sample_rate = torch.tensor(24000)
    runner = _make_runner(
        {"model_outputs": [empty], "sr": [sample_rate]},
        num_reqs=4,
    )

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.multimodal_outputs) == 4
    assert all(payload["model_outputs"].numel() == 0 for payload in output.multimodal_outputs)
    assert all(payload["sr"].item() == 24000 for payload in output.multimodal_outputs)


def test_sample_tokens_rejects_single_nonempty_output_for_batch():
    runner = _make_runner(
        {"model_outputs": [torch.ones(1)], "sr": [torch.tensor(24000)]},
        num_reqs=2,
    )

    with pytest.raises(ValueError, match="length 1 but expected 2"):
        GPUGenerationModelRunner.sample_tokens(runner)


class _StubSchedulerOutput:
    def __init__(self, total_num_scheduled_tokens):
        self.total_num_scheduled_tokens = total_num_scheduled_tokens
        self.num_scheduled_tokens = {"req-1": total_num_scheduled_tokens}
        self.finished_req_ids = set()
        self.kv_connector_metadata = None


def _make_guard_runner():
    # Stubbed far enough that a span escaping the guard reaches the real
    # `_prepare_inputs`, i.e. fails the way the reported crash does.
    runner = object.__new__(GPUGenerationModelRunner)
    runner.execute_model_state = None
    runner.routed_experts_initialized = False
    runner.speculative_config = None
    runner.model_config = SimpleNamespace(async_chunk=False)
    runner.cache_config = CacheConfig()
    runner.input_batch = _DummyInputBatch()
    runner.model = object()
    runner._update_states = lambda scheduler_output: None
    runner.synchronize_input_prep = contextlib.nullcontext
    runner.attach_omni_connector_output = lambda result: result
    return runner


@pytest.mark.parametrize("total", [-1, -512, 0])
def test_execute_model_skips_non_positive_scheduled_span(monkeypatch, total):
    """#5196: a negative span is truthy, so it used to reach `_prepare_inputs`,
    whose `assert total_num_scheduled_tokens > 0` killed the stage EngineCore."""
    monkeypatch.setattr(gen_runner_module, "has_kv_transfer_group", lambda: False)
    runner = _make_guard_runner()

    output = GPUGenerationModelRunner.execute_model(runner, _StubSchedulerOutput(total))

    assert output is EMPTY_MODEL_RUNNER_OUTPUT
