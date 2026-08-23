# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import threading
import time
from dataclasses import replace

import pytest

from vllm_omni.entrypoints.openai.aura_tool_executor import (
    AuraToolCall,
    AuraToolExecutor,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _call(
    *,
    call_id: str = "call-1",
    name: str = "mock_echo",
    arguments: dict | None = None,
) -> AuraToolCall:
    return AuraToolCall(
        id=call_id,
        name=name,
        arguments=arguments if arguments is not None else {"text": "hello"},
    )


@pytest.mark.asyncio
async def test_executor_accepts_allowlisted_strict_schema():
    executor = AuraToolExecutor()
    result = await executor.execute(
        session_id="session",
        request_id="request",
        call=_call(),
        depth=1,
    )

    assert result.status == "completed"
    assert result.content == '{"ok":true,"result":{"text":"hello"}}'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call", "depth", "error_code"),
    [
        (_call(name="not_registered"), 1, "unknown_tool"),
        (_call(arguments={"text": 123}), 1, "invalid_tool_arguments"),
        (_call(arguments={"text": "ok", "extra": True}), 1, "invalid_tool_arguments"),
        (_call(), 3, "tool_depth_exceeded"),
    ],
)
async def test_executor_fail_closed_boundaries(call, depth, error_code):
    executor = AuraToolExecutor()
    result = await executor.execute(
        session_id=f"session-{error_code}",
        request_id="request",
        call=call,
        depth=depth,
    )

    assert result.status == "error"
    assert f'"code":"{error_code}"' in result.content


@pytest.mark.asyncio
async def test_executor_deduplicates_same_call_id():
    executor = AuraToolExecutor()
    calls = 0
    spec = executor._registry["mock_echo"]

    def counted(arguments):
        nonlocal calls
        calls += 1
        return {"text": arguments.text}

    executor._registry["mock_echo"] = replace(spec, handler=counted)
    first, second = await asyncio.gather(
        executor.execute(session_id="session", request_id="request-1", call=_call(), depth=1),
        executor.execute(session_id="session", request_id="request-2", call=_call(), depth=1),
    )

    assert calls == 1
    assert first == second


@pytest.mark.asyncio
async def test_executor_timeout_and_output_cap():
    timeout_executor = AuraToolExecutor(timeout_seconds=0.01)
    spec = timeout_executor._registry["mock_echo"]

    def slow(arguments):
        time.sleep(0.1)
        return {"text": arguments.text}

    timeout_executor._registry["mock_echo"] = replace(spec, handler=slow)
    timed_out = await timeout_executor.execute(
        session_id="timeout",
        request_id="request",
        call=_call(),
        depth=1,
    )
    assert '"code":"tool_timeout"' in timed_out.content

    capped_executor = AuraToolExecutor(output_limit_bytes=16)
    capped = await capped_executor.execute(
        session_id="capped",
        request_id="request",
        call=_call(arguments={"text": "long enough to exceed the cap"}),
        depth=1,
    )
    assert '"code":"tool_output_too_large"' in capped.content


@pytest.mark.asyncio
async def test_executor_global_concurrency_limit():
    executor = AuraToolExecutor(max_concurrency=1)
    spec = executor._registry["mock_echo"]
    lock = threading.Lock()
    active = 0
    max_active = 0

    def observe(arguments):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return {"text": arguments.text}

    executor._registry["mock_echo"] = replace(spec, handler=observe)
    await asyncio.gather(
        executor.execute(session_id="s1", request_id="r1", call=_call(call_id="c1"), depth=1),
        executor.execute(session_id="s2", request_id="r2", call=_call(call_id="c2"), depth=1),
    )

    assert max_active == 1


def test_executor_from_env_selects_safe_tools(monkeypatch):
    monkeypatch.setenv("VLLM_AURA_TOOL_EXECUTOR", "safe")
    executor = AuraToolExecutor.from_env()

    assert executor is not None
    assert [schema["function"]["name"] for schema in executor.tool_schemas] == [
        "calculator",
        "get_current_datetime",
        "get_city_weather",
        "convert_currency",
        "DeepSeek",
        "get_current_location",
        "WebSearch",
    ]
    assert "mock_echo" not in executor._registry
    assert executor.force_final_after_success is True
    assert AuraToolExecutor().force_final_after_success is False

    monkeypatch.setenv("VLLM_AURA_TOOL_EXECUTOR", "off")
    assert AuraToolExecutor.from_env() is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("37 * 19", 703),
        ("sqrt(81) + 2 ** 3", 17),
        ("round(pi, 3)", 3.142),
    ],
)
async def test_safe_calculator(expression, expected):
    executor = AuraToolExecutor(mode="safe")
    result = await executor.execute(
        session_id="calculator",
        request_id=expression,
        call=_call(name="calculator", arguments={"expression": expression}),
        depth=1,
    )

    assert result.status == "completed"
    assert json.loads(result.content)["result"]["result"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "(1).__class__",
        "10 ** 101",
        "+".join(["1"] * 40),
    ],
)
async def test_safe_calculator_rejects_unsafe_or_unbounded_input(expression):
    executor = AuraToolExecutor(mode="safe")
    result = await executor.execute(
        session_id="calculator-errors",
        request_id=expression,
        call=_call(name="calculator", arguments={"expression": expression}),
        depth=1,
    )

    assert result.status == "error"


@pytest.mark.asyncio
async def test_safe_datetime_uses_shanghai_timezone():
    executor = AuraToolExecutor(mode="safe")
    result = await executor.execute(
        session_id="datetime",
        request_id="datetime",
        call=_call(name="get_current_datetime", arguments={}),
        depth=1,
    )

    payload = json.loads(result.content)["result"]
    assert payload["timezone"] == "Asia/Shanghai"
    assert payload["datetime"].endswith("+08:00")


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.content = json.dumps(payload).encode()

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_safe_weather_uses_only_fixed_open_meteo_endpoints(monkeypatch):
    calls = []
    responses = [
        {
            "results": [
                {
                    "name": "上海",
                    "country": "中國",
                    "country_code": "CN",
                    "admin1": "上海",
                    "latitude": 31.23,
                    "longitude": 121.47,
                }
            ]
        },
        {
            "timezone": "Asia/Shanghai",
            "current": {
                "time": "2026-08-20T14:00",
                "weather_code": 1,
                "temperature_2m": 31.2,
                "relative_humidity_2m": 60,
                "apparent_temperature": 35.0,
                "precipitation": 0.0,
                "wind_speed_10m": 8.0,
            },
            "current_units": {
                "temperature_2m": "°C",
                "relative_humidity_2m": "%",
                "precipitation": "mm",
                "wind_speed_10m": "km/h",
            },
        },
    ]

    def fake_get(url, *, params, timeout):
        calls.append((url, params, timeout))
        return _FakeResponse(responses.pop(0))

    monkeypatch.setattr(
        "vllm_omni.entrypoints.openai.aura_tool_executor.requests.get",
        fake_get,
    )
    executor = AuraToolExecutor(mode="safe")
    result = await executor.execute(
        session_id="weather",
        request_id="weather",
        call=_call(name="get_city_weather", arguments={"city": "上海", "country": "CN", "lang": "zh"}),
        depth=1,
    )

    payload = json.loads(result.content)["result"]
    assert result.status == "completed"
    assert payload["condition"] == "大致晴朗"
    assert [call[0] for call in calls] == [
        "https://geocoding-api.open-meteo.com/v1/search",
        "https://api.open-meteo.com/v1/forecast",
    ]


@pytest.mark.asyncio
async def test_safe_currency_validates_and_uses_fixed_frankfurter_endpoint(monkeypatch):
    calls = []

    def fake_get(url, *, params, timeout):
        calls.append((url, params, timeout))
        return _FakeResponse({"date": "2026-08-20", "rates": {"CNY": 724.5}})

    monkeypatch.setattr(
        "vllm_omni.entrypoints.openai.aura_tool_executor.requests.get",
        fake_get,
    )
    executor = AuraToolExecutor(mode="safe")
    result = await executor.execute(
        session_id="currency",
        request_id="currency",
        call=_call(
            name="convert_currency",
            arguments={"amount": 100.0, "from_currency": "usd", "to_currency": "cny"},
        ),
        depth=1,
    )

    payload = json.loads(result.content)["result"]
    assert result.status == "completed"
    assert payload["converted_amount"] == 724.5
    assert calls[0][0] == "https://api.frankfurter.app/latest"


@pytest.mark.asyncio
async def test_safe_deepseek_is_fixed_mock_without_network(monkeypatch):
    def fail_get(*_args, **_kwargs):
        raise AssertionError("DeepSeek mock must not use HTTP")

    monkeypatch.setattr(
        "vllm_omni.entrypoints.openai.aura_tool_executor.requests.get",
        fail_get,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.openai.aura_tool_executor.requests.post",
        fail_get,
    )
    executor = AuraToolExecutor(mode="safe")
    result = await executor.execute(
        session_id="deepseek",
        request_id="deepseek",
        call=_call(name="DeepSeek", arguments={"query": "今天天氣如何", "enable_search": True}),
        depth=1,
    )
    payload = json.loads(result.content)["result"]
    assert result.status == "completed"
    assert payload["mocked"] is True
    assert payload["query"] == "今天天氣如何"
    assert "真實搜尋" in payload["summary"]


@pytest.mark.asyncio
async def test_safe_location_falls_back_to_ipapi(monkeypatch):
    calls = []

    def fake_get(url, *, params=None, timeout):
        del params
        calls.append(url)
        if "ipwho.is" in url:
            return _FakeResponse({"success": False, "message": "blocked"})
        return _FakeResponse(
            {
                "ip": "1.2.3.4",
                "country_name": "中國",
                "region": "上海",
                "city": "上海",
                "latitude": 31.23,
                "longitude": 121.47,
                "timezone": "Asia/Shanghai",
                "org": "Example ISP",
            }
        )

    monkeypatch.setattr(
        "vllm_omni.entrypoints.openai.aura_tool_executor.requests.get",
        fake_get,
    )
    executor = AuraToolExecutor(mode="safe")
    result = await executor.execute(
        session_id="location",
        request_id="location",
        call=_call(name="get_current_location", arguments={}),
        depth=1,
    )
    payload = json.loads(result.content)["result"]
    assert result.status == "completed"
    assert payload["source"] == "ipapi.co"
    assert payload["city"] == "上海"
    assert calls == ["https://ipwho.is/", "https://ipapi.co/json/"]


@pytest.mark.asyncio
async def test_safe_websearch_requires_key_and_bounds_serper(monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    executor = AuraToolExecutor(mode="safe")
    missing = await executor.execute(
        session_id="search-missing",
        request_id="search-missing",
        call=_call(name="WebSearch", arguments={"query": "AURA"}),
        depth=1,
    )
    assert '"code":"search_api_key_missing"' in missing.content

    posts = []

    def fake_post(url, *, json, headers, timeout):
        posts.append((url, json, timeout))
        assert headers["X-API-KEY"] == "test-key"
        return _FakeResponse(
            {
                "organic": [
                    {"title": "Result", "snippet": "Snippet", "link": "https://example.com"}
                ]
            }
        )

    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    monkeypatch.setattr(
        "vllm_omni.entrypoints.openai.aura_tool_executor.requests.post",
        fake_post,
    )
    ok = await executor.execute(
        session_id="search-ok",
        request_id="search-ok",
        call=_call(name="WebSearch", arguments={"query": "AURA", "max_results": 3}),
        depth=1,
    )
    payload = json.loads(ok.content)["result"]
    assert ok.status == "completed"
    assert payload["source"] == "serper"
    assert payload["results"][0]["title"] == "Result"
    assert posts[0][0] == "https://google.serper.dev/search"
    assert posts[0][1] == {"q": "AURA", "num": 3}

    def huge_post(url, *, json, headers, timeout):
        del url, json, headers, timeout
        return _FakeResponse(
            {
                "organic": [
                    {"title": "x" * 400, "snippet": "y" * 400, "link": "https://example.com/" + ("z" * 400)}
                    for _ in range(20)
                ]
            }
        )

    monkeypatch.setattr(
        "vllm_omni.entrypoints.openai.aura_tool_executor.requests.post",
        huge_post,
    )
    capped = AuraToolExecutor(mode="safe", output_limit_bytes=256)
    oversized = await capped.execute(
        session_id="search-size",
        request_id="search-size",
        call=_call(name="WebSearch", arguments={"query": "big"}),
        depth=1,
    )
    assert '"code":"tool_output_too_large"' in oversized.content

    def slow_post(url, *, json, headers, timeout):
        del url, json, headers, timeout
        time.sleep(0.1)
        return _FakeResponse({"organic": []})

    monkeypatch.setattr(
        "vllm_omni.entrypoints.openai.aura_tool_executor.requests.post",
        slow_post,
    )
    timed = AuraToolExecutor(mode="safe", timeout_seconds=0.01)
    timed_out = await timed.execute(
        session_id="search-timeout",
        request_id="search-timeout",
        call=_call(name="WebSearch", arguments={"query": "slow"}),
        depth=1,
    )
    assert '"code":"tool_timeout"' in timed_out.content
