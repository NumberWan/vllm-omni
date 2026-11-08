# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Allowlisted, side-effect-free tools the AURA demo proxy runs for the browser.

Ported from the AURA_026 server-side gateway (``aura_tool_executor.py``):
same seven safe tools, same limits. ``WebSearch`` needs ``SERPER_API_KEY``;
``DeepSeek`` returns a mock without ``DEEPSEEK_API_KEY``. ``BraveSearch``,
``duckduckgo_search`` and ``WebFetch`` stay off unless
``VLLM_AURA_TOOL_BRAVE`` / ``VLLM_AURA_TOOL_DDG`` / ``VLLM_AURA_TOOL_WEBFETCH``
is set.
"""

from __future__ import annotations

import ast
import asyncio
import datetime as dt
import ipaddress
import json
import logging
import math
import operator
import os
import socket
import time
from collections.abc import Callable, Iterator
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import regex as re
import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

TOOL_TIMEOUT_SECONDS = 5.0
TOOL_OUTPUT_LIMIT_BYTES = 64 * 1024
SAFE_HTTP_TIMEOUT_SECONDS = 3.0
SAFE_HTTP_RESPONSE_LIMIT_BYTES = 256 * 1024
MAX_CALCULATOR_EXPRESSION_LENGTH = 200
MAX_CALCULATOR_AST_NODES = 64
MAX_CALCULATOR_ABS_VALUE = 1e100
MAX_CALCULATOR_EXPONENT = 100
DEEPSEEK_MAX_OUTPUT_CHARS = 8_000
DEEPSEEK_TIMEOUT_SECONDS = 30.0

_DEEPSEEK_SYSTEM_PROMPT = (
    "你是一个口语化、简洁的AI助手，必须优先使用联网搜索。"
    "默认每次回答前都先联网检索，能搜就搜，尽量不要跳过搜索。"
    "如果搜索失败，再基于已有知识给出答案，并简短说明未检索成功。"
    "用中文口语化回答，像朋友聊天一样自然。"
    "回复简短直接。不要列点、不要markdown格式。"
)
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

    @field_validator("city", "country", "lang", mode="before")
    @classmethod
    def _strip_weather_strings(cls, value: Any) -> Any:
        # Models often pad values (``lang=" zh "``); strip instead of failing the call.
        if isinstance(value, str):
            return value.strip()
        return value


class CurrencyArguments(BaseModel):
    # Not strict: AURA_v2's <parameter> bodies often arrive as strings ("1").
    model_config = ConfigDict(extra="forbid")

    amount: float = Field(ge=0, le=1e12, allow_inf_nan=False)
    from_currency: str = Field(pattern=r"^[A-Za-z]{3}$")
    to_currency: str = Field(pattern=r"^[A-Za-z]{3}$")
    date: str = Field(default="", max_length=10)

    @field_validator("amount", mode="before")
    @classmethod
    def _coerce_amount(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("amount must be a number")
        if isinstance(value, str):
            value = value.strip().replace(",", "")
        return value

    @field_validator("from_currency", "to_currency", mode="before")
    @classmethod
    def _strip_currency_code(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("date", mode="before")
    @classmethod
    def _null_date_to_empty(cls, value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return value


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


class DuckDuckGoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=256)
    max_results: int = Field(default=5, ge=1, le=10)


class WebFetchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    url: str = Field(min_length=1, max_length=2048)


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
        unary = _CALCULATOR_UNARY_OPERATORS.get(type(node.op))
        if unary is None:
            raise ValueError("calculator_unsupported_operator")
        return _bounded_number(unary(_evaluate_calculator_node(node.operand, depth=depth + 1)))
    if isinstance(node, ast.BinOp):
        binary = _CALCULATOR_BINARY_OPERATORS.get(type(node.op))
        if binary is None:
            raise ValueError("calculator_unsupported_operator")
        left = _evaluate_calculator_node(node.left, depth=depth + 1)
        right = _evaluate_calculator_node(node.right, depth=depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_CALCULATOR_EXPONENT:
            raise ValueError("calculator_exponent_out_of_range")
        return _bounded_number(binary(left, right))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _CALCULATOR_FUNCTIONS:
            raise ValueError("calculator_unsupported_function")
        if node.keywords or not 1 <= len(node.args) <= 8:
            raise ValueError("calculator_invalid_function_arguments")
        arguments: list[Any] = [_evaluate_calculator_node(arg, depth=depth + 1) for arg in node.args]
        if node.func.id == "round" and len(arguments) == 2:
            if not arguments[1].is_integer() or not -15 <= arguments[1] <= 15:
                raise ValueError("calculator_invalid_round_digits")
            arguments[1] = int(arguments[1])
        return _bounded_number(_CALCULATOR_FUNCTIONS[node.func.id](*arguments))
    raise ValueError("calculator_unsupported_expression")


def _calculator(arguments: CalculatorArguments) -> dict[str, Any]:
    expression = arguments.expression.strip()
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("calculator_invalid_syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_CALCULATOR_AST_NODES:
        raise ValueError("calculator_expression_too_complex")
    result = _evaluate_calculator_node(tree.body)
    rendered: int | float = int(result) if result.is_integer() else result
    return {"expression": expression, "result": rendered, "summary": f"{expression} = {rendered}"}


def _current_datetime(arguments: CurrentDatetimeArguments) -> dict[str, Any]:
    del arguments
    timezone = dt.timezone(dt.timedelta(hours=8), name="Asia/Shanghai")
    now = dt.datetime.now(timezone)
    return {
        "datetime": now.isoformat(timespec="seconds"),
        "timezone": "Asia/Shanghai",
        "summary": (
            f"目前上海時間是{now.year}年{now.month}月{now.day}日{now.hour}時{now.minute}分{now.second}秒（UTC+8）。"
        ),
    }


def _bounded_json(response: requests.Response) -> dict[str, Any]:
    response.raise_for_status()
    if len(response.content) > SAFE_HTTP_RESPONSE_LIMIT_BYTES:
        raise ValueError("upstream_response_too_large")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("upstream_invalid_response")
    return payload


def _get_bounded_json(url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return _bounded_json(requests.get(url, params=params, timeout=SAFE_HTTP_TIMEOUT_SECONDS))


def _post_bounded_json(
    url: str,
    *,
    json_body: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout_seconds: float = SAFE_HTTP_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    return _bounded_json(requests.post(url, json=json_body, headers=headers, timeout=timeout_seconds))


def _city_weather(arguments: CityWeatherArguments) -> dict[str, Any]:
    city = arguments.city.strip()
    country = arguments.country.strip().casefold()
    geocoding = _get_bounded_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 10, "language": arguments.lang, "format": "json"},
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
                and country in {str(item.get("country", "")).casefold(), str(item.get("country_code", "")).casefold()}
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
                "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m"
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


def _convert_currency(arguments: CurrencyArguments) -> dict[str, Any]:
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
    source_zh = _CURRENCY_NAMES_ZH.get(source, source)
    target_zh = _CURRENCY_NAMES_ZH.get(target, target)
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
                f"{arguments.amount:g} {source_zh}等於{arguments.amount:g} {target_zh}"
                f"（{arguments.amount:g} {source} = {arguments.amount:g} {target}）。"
            ),
        }
    payload = _get_bounded_json(
        f"https://api.frankfurter.app/{date or 'latest'}",
        params={"amount": arguments.amount, "from": source, "to": target},
    )
    rates = payload.get("rates")
    if not isinstance(rates, dict) or not isinstance(rates.get(target), (int, float)):
        raise ValueError("currency_rate_unavailable")
    converted = _bounded_number(rates[target])
    return {
        "amount": arguments.amount,
        "from_currency": source,
        "to_currency": target,
        "converted_amount": converted,
        "rate": round(converted / arguments.amount, 10) if arguments.amount else 0.0,
        "date": payload.get("date", date or "latest"),
        "source": "frankfurter.app",
        "summary": (
            f"{arguments.amount:g} {source_zh}約等於{converted:g} {target_zh}"
            f"（{arguments.amount:g} {source} = {converted:g} {target}）。"
        ),
    }


def _deepseek(arguments: DeepSeekArguments) -> dict[str, Any]:
    query = arguments.query.strip()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        summary = f"DeepSeek mock：已收到問題「{query}」，未呼叫真實搜尋。"
        return {"query": query, "enable_search": arguments.enable_search, "mocked": True, "summary": summary}
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/anthropic").rstrip("/")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": 256,
        "temperature": 0.7,
        "system": _DEEPSEEK_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": query}],
    }
    if arguments.enable_search:
        body["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
    try:
        payload = _post_bounded_json(
            f"{base_url}/v1/messages",
            json_body=body,
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            timeout_seconds=DEEPSEEK_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise ValueError("deepseek_upstream_failed") from exc
    content = payload.get("content")
    if not isinstance(content, list):
        raise ValueError("deepseek_invalid_response")
    parts = [
        str(block.get("text") or "").strip()
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    text = "\n".join(part for part in parts if part).strip()
    if not text:
        raise ValueError("deepseek_empty_response")
    text = text[:DEEPSEEK_MAX_OUTPUT_CHARS]
    return {"query": query, "enable_search": arguments.enable_search, "mocked": False, "model": model, "summary": text}


def _location_from_ipwho(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload.get("success", False):
        raise ValueError("location_ipwho_failed")
    timezone = payload.get("timezone")
    connection = payload.get("connection")
    return {
        "public_ip": payload.get("ip", ""),
        "country": payload.get("country", ""),
        "region": payload.get("region", ""),
        "city": payload.get("city", ""),
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "timezone": timezone.get("id", "") if isinstance(timezone, dict) else "",
        "isp": connection.get("isp", "") if isinstance(connection, dict) else "",
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


def _get_current_location(arguments: CurrentLocationArguments) -> dict[str, Any]:
    del arguments
    services: tuple[tuple[str, Callable[[dict[str, Any]], dict[str, Any]]], ...] = (
        ("https://ipwho.is/", _location_from_ipwho),
        ("https://ipapi.co/json/", _location_from_ipapi),
    )
    for url, parse in services:
        try:
            result = parse(_get_bounded_json(url))
        except (ValueError, requests.RequestException):
            continue
        place = "、".join(
            str(part) for part in (result.get("city"), result.get("region"), result.get("country")) if part
        )
        result["summary"] = f"目前大約在{place}。" if place else "已取得目前大約位置（依公開 IP 推估）。"
        return result
    raise ValueError("location_lookup_failed")


def _search_result(query: str, max_results: int, source: str, label: str, results: list[dict[str, str]]):
    summary = f"{label}「{query}」找到{len(results)}筆結果。" if results else f"{label}「{query}」沒有結果。"
    if results and results[0].get("title"):
        summary = f"{summary}首筆：{results[0]['title']}"
    return {"query": query, "max_results": max_results, "source": source, "results": results, "summary": summary}


def _web_search(arguments: WebSearchArguments) -> dict[str, Any]:
    api_key = os.environ.get("SERPER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("search_api_key_missing")
    query = arguments.query.strip()
    endpoint = os.environ.get("SERPER_SEARCH_ENDPOINT", "").strip() or "https://google.serper.dev/search"
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
    results = [
        {
            "title": str(item.get("title", ""))[:500],
            "snippet": str(item.get("snippet", ""))[:500],
            "link": str(item.get("link", ""))[:500],
        }
        for item in raw[: arguments.max_results]
        if isinstance(item, dict)
    ]
    return _search_result(query, arguments.max_results, "serper", "搜尋", results)


def _brave_search(arguments: WebSearchArguments) -> dict[str, Any]:
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
    if not api_key:
        raise ValueError("search_api_key_missing")
    query = arguments.query.strip()
    endpoint = os.environ.get("BRAVE_SEARCH_ENDPOINT", "").strip() or "https://api.search.brave.com/res/v1/web/search"
    try:
        payload = _bounded_json(
            requests.get(
                endpoint,
                headers={"Accept": "application/json", "X-Subscription-Token": api_key},
                params={"q": query, "count": arguments.max_results},
                timeout=SAFE_HTTP_TIMEOUT_SECONDS,
            )
        )
    except requests.RequestException as exc:
        raise ValueError("search_upstream_failed") from exc
    raw = (payload.get("web") or {}).get("results") or []
    if not isinstance(raw, list):
        raise ValueError("search_invalid_response")
    results = [
        {
            "title": str(item.get("title", ""))[:500],
            "snippet": str(item.get("description", ""))[:500],
            "link": str(item.get("url", ""))[:500],
        }
        for item in raw[: arguments.max_results]
        if isinstance(item, dict)
    ]
    return _search_result(query, arguments.max_results, "brave", "Brave 搜尋", results)


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._in_link = False
        self._in_snippet = False
        self._skip_snippet = False
        self._current: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        cls = attrs_dict.get("class") or ""
        if tag == "a" and "result__a" in cls:
            self._in_link = True
            self._current = {"title": "", "snippet": "", "url": str(attrs_dict.get("href") or "")}
            return
        if self._current is not None and tag == "a" and "badge" in cls:
            self._skip_snippet = True
            return
        if tag == "td" and "result__snippet" in cls:
            self._in_snippet = True

    def handle_data(self, data: str) -> None:
        if self._current is None:
            return
        if self._in_link:
            self._current["title"] += data
        elif self._in_snippet and not self._skip_snippet:
            self._current["snippet"] += data

    def handle_endtag(self, tag: str) -> None:
        if self._in_link and tag == "a":
            self._in_link = False
        if self._in_snippet and tag == "td":
            self._in_snippet = False
        if not self._in_link and not self._in_snippet and self._current is not None:
            if self._current["title"] or self._current["snippet"]:
                self.results.append(dict(self._current))
            self._current = None
            self._skip_snippet = False


def _duckduckgo_search(arguments: DuckDuckGoArguments) -> dict[str, Any]:
    query = arguments.query.strip()
    try:
        response = requests.get(
            "https://html.duckduckgo.com/html/", params={"q": query}, timeout=SAFE_HTTP_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError("search_upstream_failed") from exc
    if len(response.content) > SAFE_HTTP_RESPONSE_LIMIT_BYTES:
        raise ValueError("upstream_response_too_large")
    parser = _DuckDuckGoHTMLParser()
    parser.feed(response.text)
    results = [
        {
            "title": str(item.get("title", ""))[:500],
            "snippet": str(item.get("snippet", ""))[:500],
            "link": str(item.get("url", ""))[:500],
        }
        for item in parser.results[: arguments.max_results]
    ]
    return _search_result(query, arguments.max_results, "duckduckgo", "DuckDuckGo 搜尋", results)


def _check_ssrf(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("webfetch_invalid_url")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError("webfetch_dns_failed") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_unspecified
            or ip.is_reserved
            or ip.is_multicast
        ):
            raise ValueError("webfetch_ssrf_blocked")
    return url


def _web_fetch(arguments: WebFetchArguments) -> dict[str, Any]:
    url = _check_ssrf(arguments.url.strip().strip("\"'`“”‘’"))
    try:
        response = requests.get(
            url,
            timeout=SAFE_HTTP_TIMEOUT_SECONDS,
            allow_redirects=False,
            headers={"User-Agent": "AURA-WebFetch/1.0"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError("webfetch_upstream_failed") from exc
    if len(response.content) > SAFE_HTTP_RESPONSE_LIMIT_BYTES:
        raise ValueError("upstream_response_too_large")
    content_type = str(response.headers.get("content-type") or "")
    if any(kind in content_type for kind in ("html", "xml", "text", "json")):
        text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", response.text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()[:8000]
    else:
        text = f"Binary or non-text response ({content_type}, {len(response.content)} bytes)."
    return {"url": url, "content_type": content_type, "text": text, "summary": text[:240] if text else f"已取得 {url}"}


_SAFE_TOOL_DEFINITIONS: tuple[tuple[str, str, type[BaseModel], Callable[[Any], dict[str, Any]]], ...] = (
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
        "General assistant for open questions. Uses live DeepSeek with DEEPSEEK_API_KEY; otherwise returns a mock.",
        DeepSeekArguments,
        _deepseek,
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

_OPTIONAL_SAFE_TOOL_DEFINITIONS: tuple[tuple[str, str, str, type[BaseModel], Callable[[Any], dict[str, Any]]], ...] = (
    (
        "VLLM_AURA_TOOL_BRAVE",
        "BraveSearch",
        "Alternative web search via Brave Search API. Prefer as a second source when Serper is unavailable.",
        WebSearchArguments,
        _brave_search,
    ),
    (
        "VLLM_AURA_TOOL_DDG",
        "duckduckgo_search",
        "Search the web via DuckDuckGo HTML results. Fallback / alternative search engine.",
        DuckDuckGoArguments,
        _duckduckgo_search,
    ),
    (
        "VLLM_AURA_TOOL_WEBFETCH",
        "WebFetch",
        "Fetch a public http(s) URL and return cleaned text. Private/loopback hosts are blocked.",
        WebFetchArguments,
        _web_fetch,
    ),
)

_WEATHER_ALIASES = {
    "get_weather",
    "weather",
    "city_weather",
    "getcityweather",
    "cityweather",
    "lookup_weather",
    "查询天气",
    "查天气",
    "天气查询",
}


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _iter_tool_definitions() -> Iterator[tuple[str, str, type[BaseModel], Callable[[Any], dict[str, Any]]]]:
    yield from _SAFE_TOOL_DEFINITIONS
    for env_name, name, description, arguments_model, handler in _OPTIONAL_SAFE_TOOL_DEFINITIONS:
        if _env_enabled(env_name):
            yield name, description, arguments_model, handler


def _canonical_tool_name(name: str) -> str:
    """Map common model aliases for the weather tool onto its registered name."""
    raw = (name or "").strip()
    lowered = raw.lower().replace(" ", "_").replace("-", "_")
    collapsed = "".join(ch for ch in lowered if ch.isalnum())
    if raw in _WEATHER_ALIASES or lowered in _WEATHER_ALIASES or collapsed in _WEATHER_ALIASES:
        return "get_city_weather"
    if "weather" in collapsed and "location" not in collapsed:
        return "get_city_weather"
    return raw


def _error_output(code: str) -> str:
    return json.dumps({"ok": False, "error": {"code": code}}, separators=(",", ":"))


class AuraToolExecutor:
    """Run the allowlisted tools; one ``run`` per published function call."""

    def __init__(self) -> None:
        self._registry = {
            name: (description, arguments_model, handler)
            for name, description, arguments_model, handler in _iter_tool_definitions()
        }

    @property
    def realtime_tools(self) -> list[dict[str, Any]]:
        """Tool objects in the flat Realtime ``session.tools`` shape."""
        return [
            {
                "type": "function",
                "name": name,
                "description": description,
                "parameters": arguments_model.model_json_schema(),
            }
            for name, (description, arguments_model, _handler) in self._registry.items()
        ]

    async def run(self, name: str, arguments: str) -> str:
        """Return the ``function_call_output`` string; failures become ``{"ok": false}``."""
        started = time.perf_counter()
        tool_name = _canonical_tool_name(name)
        status = "error"
        try:
            entry = self._registry.get(tool_name)
            if entry is None:
                raise ValueError("unknown_tool")
            _description, arguments_model, handler = entry
            try:
                parsed = json.loads(arguments or "{}")
                validated = arguments_model.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError("invalid_tool_arguments") from exc
            timeout_seconds = (
                max(TOOL_TIMEOUT_SECONDS, DEEPSEEK_TIMEOUT_SECONDS)
                if tool_name == "DeepSeek" and os.environ.get("DEEPSEEK_API_KEY", "").strip()
                else TOOL_TIMEOUT_SECONDS
            )
            value = await asyncio.wait_for(asyncio.to_thread(handler, validated), timeout=timeout_seconds)
            output = json.dumps({"ok": True, "result": value}, ensure_ascii=False, separators=(",", ":"))
            if len(output.encode("utf-8")) > TOOL_OUTPUT_LIMIT_BYTES:
                raise ValueError("tool_output_too_large")
            status = "completed"
        except (TimeoutError, asyncio.TimeoutError):
            output = _error_output("tool_timeout")
        except ValueError as exc:
            output = _error_output(str(exc))
        except Exception:
            logger.exception("AURA tool %s failed", tool_name)
            output = _error_output("tool_execution_failed")
        logger.info(
            "AURA tool %s status=%s latency_ms=%.1f output_bytes=%d",
            tool_name,
            status,
            (time.perf_counter() - started) * 1000,
            len(output.encode("utf-8")),
        )
        return output
