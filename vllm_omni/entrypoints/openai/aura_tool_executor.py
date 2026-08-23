# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allowlisted, side-effect-free tool gateway for AURA_v2 streaming."""

from __future__ import annotations

import ast
import asyncio
import copy
import datetime as dt
import hashlib
import json
import math
import operator
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.logger import init_logger
from vllm.parser.qwen3 import Qwen3Parser

from vllm_omni.model_executor.stage_input_processors.aura_tool_protocol import (
    AURA_TOOL_CALL_END,
    AURA_TOOL_CALL_START,
    has_aura_tool_call_marker,
)

logger = init_logger(__name__)

DEFAULT_TOOL_TIMEOUT_SECONDS = 5.0
DEFAULT_TOOL_OUTPUT_LIMIT_BYTES = 64 * 1024
DEFAULT_TOOL_MAX_DEPTH = 2
DEFAULT_TOOL_MAX_CONCURRENCY = 8
SAFE_HTTP_TIMEOUT_SECONDS = 3.0
SAFE_HTTP_RESPONSE_LIMIT_BYTES = 256 * 1024
MAX_CALCULATOR_EXPRESSION_LENGTH = 200
MAX_CALCULATOR_AST_NODES = 64
MAX_CALCULATOR_ABS_VALUE = 1e100
MAX_CALCULATOR_EXPONENT = 100

_CURRENCY_CODE_PATTERN = re.compile(r"^[A-Z]{3}$")
_CURRENCY_NAMES_ZH = {
    "CNY": "人民幣",
    "EUR": "歐元",
    "GBP": "英鎊",
    "HKD": "港幣",
    "JPY": "日圓",
    "USD": "美元",
}
_CALCULATOR_BINARY_OPERATORS: dict[type[ast.AST], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CALCULATOR_UNARY_OPERATORS: dict[type[ast.AST], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_CALCULATOR_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
}
_CALCULATOR_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
_WEATHER_LABELS_ZH = {
    0: "晴朗",
    1: "大致晴朗",
    2: "局部多雲",
    3: "陰天",
    45: "霧",
    48: "凍霧",
    51: "小毛毛雨",
    53: "中毛毛雨",
    55: "強毛毛雨",
    56: "小凍毛毛雨",
    57: "強凍毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "小凍雨",
    67: "大凍雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "米雪",
    80: "小陣雨",
    81: "中陣雨",
    82: "強陣雨",
    85: "小陣雪",
    86: "大陣雪",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}


class MockEchoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str = Field(min_length=1, max_length=4096)


class CalculatorArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    expression: str = Field(min_length=1, max_length=MAX_CALCULATOR_EXPRESSION_LENGTH)


class CurrentDatetimeArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CityWeatherArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    city: str = Field(min_length=1, max_length=100)
    country: str = Field(default="", max_length=100)
    lang: str = Field(default="zh", pattern=r"^(zh|en)$")


class CurrencyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    amount: float = Field(ge=0, le=1e12, allow_inf_nan=False)
    from_currency: str = Field(pattern=r"^[A-Za-z]{3}$")
    to_currency: str = Field(pattern=r"^[A-Za-z]{3}$")
    date: str = Field(default="", max_length=10)


class DeepSeekArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=4096)
    enable_search: bool = True


class CurrentLocationArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class WebSearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=256)
    max_results: int = Field(default=10, ge=1, le=20)


@dataclass(frozen=True)
class AuraToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ParsedAuraToolTurn:
    reasoning: str | None
    content: str | None
    calls: list[AuraToolCall]
    error: str | None = None


@dataclass(frozen=True)
class AuraToolResult:
    call_id: str
    name: str
    status: str
    content: str
    latency_ms: float
    output_bytes: int


@dataclass(frozen=True)
class _ToolSpec:
    schema: dict[str, Any]
    arguments_model: type[BaseModel]
    handler: Callable[[BaseModel], Any]


def _mock_echo(arguments: BaseModel) -> dict[str, Any]:
    assert isinstance(arguments, MockEchoArguments)
    return {"text": arguments.text}


_MOCK_ECHO_SCHEMA = {
    "type": "function",
    "function": {
        "name": "mock_echo",
        "description": "Return the supplied text unchanged. This deterministic mock has no side effects.",
        "parameters": MockEchoArguments.model_json_schema(),
    },
}


def _bounded_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("calculator_non_numeric_result")
    result = float(value)
    if not math.isfinite(result) or abs(result) > MAX_CALCULATOR_ABS_VALUE:
        raise ValueError("calculator_result_out_of_range")
    return result


def _evaluate_calculator_node(node: ast.AST, *, depth: int = 0) -> float:
    if depth > 12:
        raise ValueError("calculator_expression_too_deep")
    if isinstance(node, ast.Constant):
        return _bounded_number(node.value)
    if isinstance(node, ast.Name):
        if node.id not in _CALCULATOR_CONSTANTS:
            raise ValueError("calculator_unknown_name")
        return _CALCULATOR_CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp):
        operation = _CALCULATOR_UNARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("calculator_unsupported_operator")
        return _bounded_number(operation(_evaluate_calculator_node(node.operand, depth=depth + 1)))
    if isinstance(node, ast.BinOp):
        operation = _CALCULATOR_BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("calculator_unsupported_operator")
        left = _evaluate_calculator_node(node.left, depth=depth + 1)
        right = _evaluate_calculator_node(node.right, depth=depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_CALCULATOR_EXPONENT:
            raise ValueError("calculator_exponent_out_of_range")
        return _bounded_number(operation(left, right))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _CALCULATOR_FUNCTIONS:
            raise ValueError("calculator_unsupported_function")
        if node.keywords or not 1 <= len(node.args) <= 8:
            raise ValueError("calculator_invalid_function_arguments")
        arguments = [_evaluate_calculator_node(arg, depth=depth + 1) for arg in node.args]
        if node.func.id == "round" and len(arguments) == 2:
            if not arguments[1].is_integer() or not -15 <= arguments[1] <= 15:
                raise ValueError("calculator_invalid_round_digits")
            arguments[1] = int(arguments[1])
        return _bounded_number(_CALCULATOR_FUNCTIONS[node.func.id](*arguments))
    raise ValueError("calculator_unsupported_expression")


def _calculator(arguments: BaseModel) -> dict[str, Any]:
    assert isinstance(arguments, CalculatorArguments)
    expression = arguments.expression.strip()
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("calculator_invalid_syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_CALCULATOR_AST_NODES:
        raise ValueError("calculator_expression_too_complex")
    result = _evaluate_calculator_node(tree.body)
    rendered_result: int | float = int(result) if result.is_integer() else result
    return {
        "expression": expression,
        "result": rendered_result,
        "summary": f"{expression} = {rendered_result}",
    }


def _current_datetime(arguments: BaseModel) -> dict[str, Any]:
    assert isinstance(arguments, CurrentDatetimeArguments)
    timezone = dt.timezone(dt.timedelta(hours=8), name="Asia/Shanghai")
    now = dt.datetime.now(timezone)
    return {
        "datetime": now.isoformat(timespec="seconds"),
        "timezone": "Asia/Shanghai",
        "summary": (
            f"目前上海時間是{now.year}年{now.month}月{now.day}日"
            f"{now.hour}時{now.minute}分{now.second}秒（UTC+8）。"
        ),
    }


def _get_bounded_json(url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(
        url,
        params=params,
        timeout=SAFE_HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    if len(response.content) > SAFE_HTTP_RESPONSE_LIMIT_BYTES:
        raise ValueError("upstream_response_too_large")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("upstream_invalid_response")
    return payload


def _post_bounded_json(
    url: str,
    *,
    json_body: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    response = requests.post(
        url,
        json=json_body,
        headers=headers,
        timeout=SAFE_HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    if len(response.content) > SAFE_HTTP_RESPONSE_LIMIT_BYTES:
        raise ValueError("upstream_response_too_large")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("upstream_invalid_response")
    return payload


def _city_weather(arguments: BaseModel) -> dict[str, Any]:
    assert isinstance(arguments, CityWeatherArguments)
    city = arguments.city.strip()
    country = arguments.country.strip().casefold()
    geocoding = _get_bounded_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": city,
            "count": 10,
            "language": arguments.lang,
            "format": "json",
        },
    )
    candidates = geocoding.get("results")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("weather_city_not_found")
    location = candidates[0]
    if country:
        location = next(
            (
                item
                for item in candidates
                if isinstance(item, dict)
                and country
                in {
                    str(item.get("country", "")).casefold(),
                    str(item.get("country_code", "")).casefold(),
                }
            ),
            location,
        )
    if not isinstance(location, dict):
        raise ValueError("weather_invalid_location")
    latitude = location.get("latitude")
    longitude = location.get("longitude")
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        raise ValueError("weather_missing_coordinates")
    weather = _get_bounded_json(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "precipitation,weather_code,wind_speed_10m"
            ),
            "timezone": "auto",
        },
    )
    current = weather.get("current")
    units = weather.get("current_units")
    if not isinstance(current, dict):
        raise ValueError("weather_current_unavailable")
    if not isinstance(units, dict):
        units = {}
    code = int(current.get("weather_code", -1))
    result = {
        "city": location.get("name", city),
        "country": location.get("country", ""),
        "region": location.get("admin1", ""),
        "timezone": weather.get("timezone", ""),
        "time": current.get("time", ""),
        "condition": _WEATHER_LABELS_ZH.get(code, f"未知天氣代碼 {code}"),
        "temperature": current.get("temperature_2m"),
        "temperature_unit": units.get("temperature_2m", ""),
        "feels_like": current.get("apparent_temperature"),
        "humidity": current.get("relative_humidity_2m"),
        "humidity_unit": units.get("relative_humidity_2m", ""),
        "precipitation": current.get("precipitation"),
        "precipitation_unit": units.get("precipitation", ""),
        "wind_speed": current.get("wind_speed_10m"),
        "wind_speed_unit": units.get("wind_speed_10m", ""),
        "source": "open-meteo.com",
    }
    result["summary"] = (
        f"{result['city']}目前{result['condition']}，氣溫{result['temperature']}"
        f"{result['temperature_unit']}，體感{result['feels_like']}{result['temperature_unit']}。"
    )
    return result


def _convert_currency(arguments: BaseModel) -> dict[str, Any]:
    assert isinstance(arguments, CurrencyArguments)
    source = arguments.from_currency.upper()
    target = arguments.to_currency.upper()
    if not _CURRENCY_CODE_PATTERN.fullmatch(source) or not _CURRENCY_CODE_PATTERN.fullmatch(target):
        raise ValueError("currency_invalid_code")
    date = arguments.date.strip()
    if date:
        try:
            dt.date.fromisoformat(date)
        except ValueError as exc:
            raise ValueError("currency_invalid_date") from exc
    if source == target:
        return {
            "amount": arguments.amount,
            "from_currency": source,
            "to_currency": target,
            "converted_amount": arguments.amount,
            "rate": 1.0,
            "date": date or "latest",
            "source": "identity",
            "summary": (
                f"{arguments.amount:g} {_CURRENCY_NAMES_ZH.get(source, source)}"
                f"等於{arguments.amount:g} {_CURRENCY_NAMES_ZH.get(target, target)}"
                f"（{arguments.amount:g} {source} = {arguments.amount:g} {target}）。"
            ),
        }
    payload = _get_bounded_json(
        f"https://api.frankfurter.app/{date or 'latest'}",
        params={
            "amount": arguments.amount,
            "from": source,
            "to": target,
        },
    )
    rates = payload.get("rates")
    if not isinstance(rates, dict) or not isinstance(rates.get(target), (int, float)):
        raise ValueError("currency_rate_unavailable")
    converted = _bounded_number(rates[target])
    result = {
        "amount": arguments.amount,
        "from_currency": source,
        "to_currency": target,
        "converted_amount": converted,
        "rate": round(converted / arguments.amount, 10) if arguments.amount else 0.0,
        "date": payload.get("date", date or "latest"),
        "source": "frankfurter.app",
    }
    result["summary"] = (
        f"{arguments.amount:g} {_CURRENCY_NAMES_ZH.get(source, source)}"
        f"約等於{converted:g} {_CURRENCY_NAMES_ZH.get(target, target)}"
        f"（{arguments.amount:g} {source} = {converted:g} {target}）。"
    )
    return result


def _deepseek_mock(arguments: BaseModel) -> dict[str, Any]:
    assert isinstance(arguments, DeepSeekArguments)
    query = arguments.query.strip()
    summary = f"DeepSeek mock：已收到問題「{query}」，未呼叫真實搜尋。"
    return {
        "query": query,
        "enable_search": arguments.enable_search,
        "mocked": True,
        "text": summary,
        "summary": summary,
    }


def _location_from_ipwho(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload.get("success", False):
        raise ValueError("location_ipwho_failed")
    timezone = payload.get("timezone")
    timezone_id = timezone.get("id", "") if isinstance(timezone, dict) else ""
    connection = payload.get("connection")
    isp = connection.get("isp", "") if isinstance(connection, dict) else ""
    return {
        "public_ip": payload.get("ip", ""),
        "country": payload.get("country", ""),
        "region": payload.get("region", ""),
        "city": payload.get("city", ""),
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "timezone": timezone_id,
        "isp": isp,
        "source": "ipwho.is",
        "accuracy_note": "Approximate location inferred from public IP.",
    }


def _location_from_ipapi(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("error"):
        raise ValueError("location_ipapi_failed")
    return {
        "public_ip": payload.get("ip", ""),
        "country": payload.get("country_name", ""),
        "region": payload.get("region", ""),
        "city": payload.get("city", ""),
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "timezone": payload.get("timezone", ""),
        "isp": payload.get("org", ""),
        "source": "ipapi.co",
        "accuracy_note": "Approximate location inferred from public IP.",
    }


def _get_current_location(arguments: BaseModel) -> dict[str, Any]:
    assert isinstance(arguments, CurrentLocationArguments)
    del arguments
    services = (
        ("https://ipwho.is/", _location_from_ipwho),
        ("https://ipapi.co/json/", _location_from_ipapi),
    )
    for url, parser in services:
        try:
            result = parser(_get_bounded_json(url))
        except (ValueError, requests.RequestException):
            continue
        place = "、".join(
            str(part)
            for part in (result.get("city"), result.get("region"), result.get("country"))
            if part
        )
        result["summary"] = (
            f"目前大約在{place}。" if place else "已取得目前大約位置（依公開 IP 推估）。"
        )
        return result
    raise ValueError("location_lookup_failed")


def _web_search(arguments: BaseModel) -> dict[str, Any]:
    assert isinstance(arguments, WebSearchArguments)
    api_key = os.environ.get("SERPER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("search_api_key_missing")
    query = arguments.query.strip()
    endpoint = os.environ.get("SERPER_SEARCH_ENDPOINT", "https://google.serper.dev/search").strip()
    if not endpoint:
        endpoint = "https://google.serper.dev/search"
    try:
        payload = _post_bounded_json(
            endpoint,
            json_body={"q": query, "num": arguments.max_results},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        )
    except requests.RequestException as exc:
        raise ValueError("search_upstream_failed") from exc
    raw = payload.get("organic") or []
    if not isinstance(raw, list):
        raise ValueError("search_invalid_response")
    results: list[dict[str, str]] = []
    for item in raw[: arguments.max_results]:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": str(item.get("title", ""))[:500],
                "snippet": str(item.get("snippet", ""))[:500],
                "link": str(item.get("link", ""))[:500],
            }
        )
    summary = (
        f"搜尋「{query}」找到{len(results)}筆結果。"
        if results
        else f"搜尋「{query}」沒有結果。"
    )
    if results and results[0].get("title"):
        summary = f"{summary}首筆：{results[0]['title']}"
    return {
        "query": query,
        "max_results": arguments.max_results,
        "source": "serper",
        "results": results,
        "summary": summary,
    }


def _tool_schema(name: str, description: str, arguments_model: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": arguments_model.model_json_schema(),
        },
    }


_SAFE_TOOL_DEFINITIONS = (
    (
        "calculator",
        "Safely evaluate arithmetic. Treat its result as final; do not call another tool to verify it.",
        CalculatorArguments,
        _calculator,
    ),
    (
        "get_current_datetime",
        "Return the current Asia/Shanghai date and time. Treat its result as final and answer without other tools.",
        CurrentDatetimeArguments,
        _current_datetime,
    ),
    (
        "get_city_weather",
        "Look up current city weather using Open-Meteo. Treat its result as final; do not call another tool.",
        CityWeatherArguments,
        _city_weather,
    ),
    (
        "convert_currency",
        "Convert an amount between ISO currencies. The converted amount is final; do not recalculate it.",
        CurrencyArguments,
        _convert_currency,
    ),
    (
        "DeepSeek",
        "General assistant for open questions. This Omni deployment returns a fixed mock and does not call a live model.",
        DeepSeekArguments,
        _deepseek_mock,
    ),
    (
        "get_current_location",
        "Get the approximate geolocation from the public IP. Treat the result as final.",
        CurrentLocationArguments,
        _get_current_location,
    ),
    (
        "WebSearch",
        "General-purpose web search. Use for factual lookup and recent events. Treat ranked results as final.",
        WebSearchArguments,
        _web_search,
    ),
)


def parse_aura_tool_output(
    tokenizer: Any,
    raw_output: str,
    *,
    request_id: str,
    tool_schemas: list[dict[str, Any]],
) -> ParsedAuraToolTurn:
    """Parse complete Qwen3.5 XML output and assign stable server call IDs."""

    has_tool_marker = has_aura_tool_call_marker(raw_output)
    if has_tool_marker and raw_output.count(AURA_TOOL_CALL_START) != raw_output.count(AURA_TOOL_CALL_END):
        return ParsedAuraToolTurn(None, None, [], "malformed_tool_xml")

    request = ChatCompletionRequest(
        model="AURA_v2",
        messages=[{"role": "user", "content": "server-side tool transaction"}],
        tools=tool_schemas,
    )
    try:
        parser = Qwen3Parser(tokenizer, request.tools)
        reasoning, content, parsed_calls = parser.parse(
            raw_output,
            request,
            enable_auto_tools=True,
            model_output_token_ids=tokenizer.encode(raw_output, add_special_tokens=False),
        )
    except Exception:
        logger.warning("AURA tool XML parse failed request_id=%s", request_id, exc_info=True)
        if has_tool_marker:
            return ParsedAuraToolTurn(None, None, [], "malformed_tool_xml")
        return ParsedAuraToolTurn(None, raw_output, [])

    if not parsed_calls:
        if has_tool_marker:
            return ParsedAuraToolTurn(reasoning, content, [], "malformed_tool_xml")
        natural_content = content if content is not None else raw_output
        return ParsedAuraToolTurn(reasoning, natural_content.lstrip(), [])

    calls: list[AuraToolCall] = []
    for index, parsed in enumerate(parsed_calls):
        try:
            arguments = json.loads(parsed.arguments)
        except (TypeError, json.JSONDecodeError):
            return ParsedAuraToolTurn(reasoning, content, [], "malformed_tool_arguments")
        if not isinstance(arguments, dict):
            return ParsedAuraToolTurn(reasoning, content, [], "malformed_tool_arguments")
        digest = hashlib.sha256(f"{request_id}:{index}:{parsed.name}:{parsed.arguments}".encode()).hexdigest()[:16]
        calls.append(
            AuraToolCall(
                id=f"call_{digest}",
                name=parsed.name,
                arguments=arguments,
            )
        )
    return ParsedAuraToolTurn(reasoning, content, calls)


class AuraToolExecutor:
    """Execute fixed allowlisted tools behind strict resource limits."""

    def __init__(
        self,
        *,
        mode: str = "mock",
        timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
        output_limit_bytes: int = DEFAULT_TOOL_OUTPUT_LIMIT_BYTES,
        max_depth: int = DEFAULT_TOOL_MAX_DEPTH,
        max_concurrency: int = DEFAULT_TOOL_MAX_CONCURRENCY,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.output_limit_bytes = output_limit_bytes
        self.max_depth = max_depth
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._dedupe_lock = asyncio.Lock()
        self._tasks: dict[tuple[str, str], asyncio.Task[AuraToolResult]] = {}
        self.force_final_after_success = mode == "safe"
        if mode == "mock":
            self._registry = {
                "mock_echo": _ToolSpec(
                    schema=_MOCK_ECHO_SCHEMA,
                    arguments_model=MockEchoArguments,
                    handler=_mock_echo,
                )
            }
        elif mode == "safe":
            self._registry = {
                name: _ToolSpec(
                    schema=_tool_schema(name, description, arguments_model),
                    arguments_model=arguments_model,
                    handler=handler,
                )
                for name, description, arguments_model, handler in _SAFE_TOOL_DEFINITIONS
            }
        else:
            raise ValueError("AuraToolExecutor mode must be 'mock' or 'safe'")

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(spec.schema) for spec in self._registry.values()]

    @classmethod
    def from_env(cls) -> AuraToolExecutor | None:
        mode = os.environ.get("VLLM_AURA_TOOL_EXECUTOR", "").strip().lower()
        if mode in {"", "0", "off", "none", "false"}:
            return None
        if mode not in {"mock", "safe"}:
            raise ValueError("VLLM_AURA_TOOL_EXECUTOR must be 'mock', 'safe', or disabled")
        return cls(mode=mode)

    async def execute(
        self,
        *,
        session_id: str,
        request_id: str,
        call: AuraToolCall,
        depth: int,
    ) -> AuraToolResult:
        """Execute a call exactly once per session/call ID."""

        key = (session_id, call.id)
        async with self._dedupe_lock:
            task = self._tasks.get(key)
            if task is None:
                task = asyncio.create_task(
                    self._execute_once(
                        session_id=session_id,
                        request_id=request_id,
                        call=call,
                        depth=depth,
                    )
                )
                self._tasks[key] = task
        return await asyncio.shield(task)

    async def _execute_once(
        self,
        *,
        session_id: str,
        request_id: str,
        call: AuraToolCall,
        depth: int,
    ) -> AuraToolResult:
        started = time.perf_counter()
        status = "error"
        content = ""
        try:
            if depth < 1 or depth > self.max_depth:
                raise ValueError("tool_depth_exceeded")
            spec = self._registry.get(call.name)
            if spec is None:
                raise ValueError("unknown_tool")
            try:
                validated = spec.arguments_model.model_validate(call.arguments)
            except ValidationError as exc:
                raise ValueError("invalid_tool_arguments") from exc

            async with self._semaphore:
                value = await asyncio.wait_for(
                    asyncio.to_thread(spec.handler, validated),
                    timeout=self.timeout_seconds,
                )
            content = json.dumps(
                {"ok": True, "result": value},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if len(content.encode("utf-8")) > self.output_limit_bytes:
                raise ValueError("tool_output_too_large")
            status = "completed"
        except asyncio.TimeoutError:
            content = self._error_content("tool_timeout")
        except ValueError as exc:
            content = self._error_content(str(exc))
        except Exception:
            logger.exception(
                "AURA tool execution failed request_id=%s session_id=%s call_id=%s tool=%s",
                request_id,
                session_id,
                call.id,
                call.name,
            )
            content = self._error_content("tool_execution_failed")

        latency_ms = (time.perf_counter() - started) * 1000
        output_bytes = len(content.encode("utf-8"))
        logger.info(
            "AURA tool audit request_id=%s session_id=%s call_id=%s tool=%s "
            "latency_ms=%.3f status=%s output_bytes=%d",
            request_id,
            session_id,
            call.id,
            call.name,
            latency_ms,
            status,
            output_bytes,
        )
        return AuraToolResult(
            call_id=call.id,
            name=call.name,
            status=status,
            content=content,
            latency_ms=latency_ms,
            output_bytes=output_bytes,
        )

    def clear_session(self, session_id: str) -> None:
        for key in [key for key in self._tasks if key[0] == session_id]:
            task = self._tasks.pop(key)
            if not task.done():
                task.cancel()

    @staticmethod
    def _error_content(code: str) -> str:
        return json.dumps(
            {"ok": False, "error": {"code": code}},
            separators=(",", ":"),
        )
