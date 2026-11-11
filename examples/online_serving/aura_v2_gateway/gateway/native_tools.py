# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The six tools the native AURA demo exposes, with its exact schemas.

``_DEFINITIONS`` is the native gateway's ``get_tool_definitions()`` output
under the demo model's saved tool profile (the five ``config.py`` defaults
plus ``deepseek_vlm_search``), kept verbatim and in registry order. Looser
descriptions or extra search tools make
the model call search for requests it should answer from the frame, such
as drawing a box.

Execution follows the native ``tools/`` package and ``ToolRunner``: each
tool returns plain text, which goes back to the model unwrapped. Errors are
plain text too (``tool not found: <name>``, ``tool timeout``, the exception
message).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import time
from collections.abc import Callable
from typing import Any

import requests

_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "deepseek_vlm_search",
            "description": "结合当前画面，检索画面中清晰可见对象需要联网或外部知识才能确认的信息，例如公开名称、身份和关键背景资料。系统会把发起调用那一刻的当前画面自动当作输入，无需也不要传图片参数。前提：必须等画面中的目标已经清晰可见、对焦稳定后再调用；目标模糊、晃动、被遮挡或尚未对准时禁止调用，先等待画面清晰。仅在回答必须依赖这类额外网络知识时才可调用；可凭自身视觉直接回答时禁止调用。返回结果仅供参考，以你自己的视觉判断为准。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "需要对照当前清晰画面才能对应到的外部或实时信息问题。目标尚未清晰可见时不要调用；禁止写成「帮我看清」之类的纯视觉问题。",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "从网络时间服务读取当前北京日期和时间，返回带时区的 ISO 8601。仅在回答必须依赖当前真实日期或时刻这一实时信息时才可调用，例如用户明确问现在几点、今天几号。禁止因为画面看不清、闲聊或可凭常识回答而调用。返回结果仅供参考，以你自己的判断为准。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "DeepSeek",
            "description": "按文本在网上检索外部知识或实时信息。不看画面、也不识图，只接受文字问题。只在回答必须依赖联网文字资料时使用；自己已经知道的不必查。结果仅供参考。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要检索的文本问题。只接受文字，不要写入需要看图才能问的内容。",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_location",
            "description": "根据公网 IP 查询大致地理位置（城市、地区、国家等），仅城市级精度，无法精确到街道、建筑或画面中的地点。仅在回答必须依赖用户当前所在城市或地区这一额外信息时才可调用。禁止用本工具判断画面里是什么地方——地点外观必须依赖你自己的视觉能力。不要因为看不清或想确认自己的判断而调用。返回结果仅供参考，以你自己的判断为准。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_city_weather",
            "description": "查询指定城市的实时或指定日期天气，返回温度、天气状况、降水、风速等实时气象数据。仅在用户明确询问天气、气温、降水等且必须依赖这类实时信息时才可调用。禁止主动或顺便查询；画面中的天色、是否在下雨必须先用你自己的视觉判断。用户未给出城市时，需要城市名称才能查询。返回结果仅供参考，以你自己的判断为准。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. Shanghai, Singapore, Paris."},
                    "country": {"type": "string", "description": "Optional country name/code to disambiguate."},
                    "lang": {"type": "string", "description": "Language hint, e.g. zh or en. Default zh."},
                    "date": {"type": "string", "description": "Optional date in YYYY-MM-DD format. 不传则查实时天气。"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visual_web_search",
            "description": "辅助检索：把当前最新画面和文本问题一起提交到互联网搜索，返回一段文本答案。调用时自动附带那一刻的画面，不必也不要传入图片。等目标清晰可见、对焦稳定后再调用。画面模糊、晃动、被遮挡或尚未对准时不要调用，先等画面清楚。只在答案取决于画面中的对象，且该事实必须上网才能确认时调用，例如公开名称、身份、地点、商品或实时背景。凭自己看图或已有知识就能回答时不要调用。结果仅供参考，画面仍以你自己看到的为准。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "连同当前清晰画面一起上网检索的具体问题。写明要确认的外部事实，例如「画面中这家店的公开名称和主营」。目标尚未清晰入画时不要调用。不要写成只要求看清画面的问题。",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

_BEIJING = dt.timezone(dt.timedelta(hours=8))
_DEEPSEEK_URL = "https://api.deepseek.com/anthropic/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_DEEPSEEK_CHAT_URL = "https://api.deepseek.com/chat/completions"

_WEATHER_CODES = {
    0: ("Clear sky", "晴朗"),
    1: ("Mainly clear", "大体晴朗"),
    2: ("Partly cloudy", "局部多云"),
    3: ("Overcast", "阴天"),
    45: ("Fog", "雾"),
    48: ("Depositing rime fog", "冻雾"),
    51: ("Light drizzle", "小毛毛雨"),
    53: ("Moderate drizzle", "中毛毛雨"),
    55: ("Dense drizzle", "强毛毛雨"),
    56: ("Light freezing drizzle", "小冻毛毛雨"),
    57: ("Dense freezing drizzle", "强冻毛毛雨"),
    61: ("Slight rain", "小雨"),
    63: ("Moderate rain", "中雨"),
    65: ("Heavy rain", "大雨"),
    66: ("Light freezing rain", "小冻雨"),
    67: ("Heavy freezing rain", "强冻雨"),
    71: ("Slight snow fall", "小雪"),
    73: ("Moderate snow fall", "中雪"),
    75: ("Heavy snow fall", "大雪"),
    77: ("Snow grains", "米雪"),
    80: ("Slight rain showers", "小阵雨"),
    81: ("Moderate rain showers", "中阵雨"),
    82: ("Violent rain showers", "强阵雨"),
    85: ("Slight snow showers", "小阵雪"),
    86: ("Heavy snow showers", "强阵雪"),
    95: ("Thunderstorm", "雷暴"),
    96: ("Thunderstorm with slight hail", "伴有小冰雹的雷暴"),
    99: ("Thunderstorm with heavy hail", "伴有强冰雹的雷暴"),
}

_DEEPSEEK_PROMPT = (
    "你是联网文字检索助手。针对用户的文本问题，检索并回答需要上网才能确认的事实或实时信息。"
    "用中文简短作答，不要列点、不要 markdown。"
    "查不到或无法确认就直说，不要编造。"
)
_VISUAL_PROMPT = (
    "你是通用图搜助手。你会同时看到当前画面和一条检索问题。"
    "先确认问题指向画面里哪一个清晰可见的对象，再联网检索这个对象需要外部资料才能确定的事实，并据此作答。"
    "用中文简洁准确回答，不要列点，不要 markdown。"
    "看不清、问题和画面对不上、或网上查不到时直接说明。不要编造，也不要只描写外观。"
)
_VLM_PROMPT = (
    "你是通用图搜助手。结合当前画面，检索画面中清晰可见对象需要联网或外部知识才能确认的信息，"
    "例如公开名称、身份和关键背景资料。"
    "用中文简洁准确作答。看不清或无法确认的细节不要编造，也不要只复述外观。"
)


def _dated_prompt(prompt: str) -> str:
    now = dt.datetime.now(_BEIJING)
    return f"当前日期时间：{now.strftime('%Y-%m-%d %H:%M')}（北京时间）\n\n{prompt}"


def _deepseek_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise ValueError("DeepSeek API key is not configured")
    return key


def _anthropic_text(data: dict[str, Any]) -> str:
    parts = [
        str(block.get("text") or "").strip()
        for block in data.get("content") or []
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(part for part in parts if part).strip()


def _request_error_detail(exc: requests.RequestException) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    body = (response.text or "").strip()
    if len(body) > 500:
        body = body[:500] + "..."
    return f"{exc}; body={body}" if body else str(exc)


def get_current_datetime() -> str:
    try:
        resp = requests.get("https://worldtimeapi.org/api/timezone/Asia/Shanghai", timeout=5)
        resp.raise_for_status()
        return resp.json()["datetime"]
    except Exception:
        return dt.datetime.now(_BEIJING).isoformat(timespec="seconds")


def ask_deepseek(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "Error: query is required"
    resp = requests.post(
        _DEEPSEEK_URL,
        headers={"x-api-key": _deepseek_key(), "anthropic-version": _ANTHROPIC_VERSION},
        json={
            "model": "deepseek-chat",
            "max_tokens": 256,
            "temperature": 0.7,
            "system": _dated_prompt(_DEEPSEEK_PROMPT),
            "messages": [{"role": "user", "content": q}],
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    text = _anthropic_text(resp.json())
    if not text:
        return "DeepSeek returned no text content."
    if len(text) > 4096:
        text = text[:4096].rstrip() + "…"
    return text


def _from_ipwho(payload: dict) -> dict:
    if not payload.get("success", False):
        raise ValueError(payload.get("message", "ipwho.is lookup failed"))
    return {
        "public_ip": payload.get("ip", ""),
        "country": payload.get("country", ""),
        "region": payload.get("region", ""),
        "city": payload.get("city", ""),
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "timezone": payload.get("timezone", {}).get("id", ""),
        "isp": payload.get("connection", {}).get("isp", ""),
        "source": "ipwho.is",
        "accuracy_note": "Approximate location inferred from public IP.",
    }


def _from_ipapi(payload: dict) -> dict:
    if payload.get("error"):
        raise ValueError(payload.get("reason", "ipapi.co lookup failed"))
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


def get_current_location() -> str:
    errors: list[str] = []
    for url, parser in (("https://ipwho.is/", _from_ipwho), ("https://ipapi.co/json/", _from_ipapi)):
        try:
            resp = requests.get(url, timeout=6)
            resp.raise_for_status()
            return json.dumps(parser(resp.json()), ensure_ascii=False)
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    return "Location lookup failed: " + " | ".join(errors)


def _weather_label(code: int, lang: str) -> str:
    info = _WEATHER_CODES.get(code)
    if info is None:
        return "Unknown weather"
    return info[1] if lang.lower().startswith("zh") else info[0]


def get_city_weather(city: str, country: str = "", lang: str = "zh", date: str = "") -> str:
    city_name = (city or "").strip()
    country_name = (country or "").strip().lower()
    language = (lang or "zh").strip() or "zh"
    date_str = (date or "").strip()
    if not city_name:
        return "Error: city is required"
    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city_name, "count": 10, "language": language, "format": "json"},
            timeout=10,
        )
        geo_resp.raise_for_status()
        candidates = geo_resp.json().get("results") or []
        if not candidates:
            return f"Weather lookup failed: city not found ({city_name})"
        location = candidates[0]
        if country_name:
            for item in candidates:
                if country_name in {str(item.get("country", "")).lower(), str(item.get("country_code", "")).lower()}:
                    location = item
                    break
        lat = location.get("latitude")
        lon = location.get("longitude")
        if lat is None or lon is None:
            return "Weather lookup failed: missing coordinates"

        if date_str:
            try:
                query_date = dt.date.fromisoformat(date_str)
            except ValueError:
                return "日期格式错误，请使用 YYYY-MM-DD"
            api_base = (
                "https://api.open-meteo.com/v1/forecast"
                if query_date > dt.date.today()
                else "https://archive-api.open-meteo.com/v1/archive"
            )
            weather_resp = requests.get(
                api_base,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_sum,wind_speed_10m_max",
                    "start_date": date_str,
                    "end_date": date_str,
                    "timezone": "auto",
                },
                timeout=10,
            )
            weather_resp.raise_for_status()
            weather_data = weather_resp.json()
            daily = weather_data.get("daily") or {}
            units = weather_data.get("daily_units") or {}
            if not daily.get("time", []):
                return f"未找到 {date_str} 的天气数据"
            code = int((daily.get("weather_code") or [None])[0] or -1)
            return json.dumps(
                {
                    "city": location.get("name", city_name),
                    "country": location.get("country", ""),
                    "region": location.get("admin1", ""),
                    "latitude": lat,
                    "longitude": lon,
                    "timezone": weather_data.get("timezone", ""),
                    "date": date_str,
                    "weather_code": code,
                    "weather_text": _weather_label(code, language),
                    "temperature_max": (daily.get("temperature_2m_max") or [None])[0],
                    "temperature_min": (daily.get("temperature_2m_min") or [None])[0],
                    "temperature_unit": units.get("temperature_2m_max", ""),
                    "precipitation": (daily.get("precipitation_sum") or [None])[0],
                    "precipitation_unit": units.get("precipitation_sum", ""),
                    "wind_speed": (daily.get("wind_speed_10m_max") or [None])[0],
                    "wind_speed_unit": units.get("wind_speed_10m_max", ""),
                    "source": "open-meteo.com",
                },
                ensure_ascii=False,
            )

        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m"
                ),
                "timezone": "auto",
            },
            timeout=10,
        )
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()
        current = weather_data.get("current") or {}
        units = weather_data.get("current_units") or {}
        code = int(current.get("weather_code", -1))
        return json.dumps(
            {
                "city": location.get("name", city_name),
                "country": location.get("country", ""),
                "region": location.get("admin1", ""),
                "latitude": lat,
                "longitude": lon,
                "timezone": weather_data.get("timezone", ""),
                "time": current.get("time", ""),
                "weather_code": code,
                "weather_text": _weather_label(code, language),
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
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return f"Weather lookup failed: {exc}"


def visual_web_search(query: str, image: str = "") -> str:
    q = (query or "").strip()
    if not q:
        return "visual_web_search failed: query is required."
    if not (image or "").strip():
        return (
            "visual_web_search failed: no video frame or image has been received yet, so there is no image to search."
        )
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return "visual_web_search failed: configure DeepSeek API key in gateway/tools/config.py."
    payload = {
        "model": os.environ.get("DEEPSEEK_VISION_MODEL", "deepseek-flash"),
        "max_tokens": 1024,
        "thinking": {"type": "disabled"},
        "system": _dated_prompt(_VISUAL_PROMPT),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": q},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image}},
                ],
            }
        ],
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
    }
    data: Any = None
    for attempt in range(3):
        try:
            response = requests.post(
                _DEEPSEEK_URL,
                headers={"x-api-key": api_key, "anthropic-version": _ANTHROPIC_VERSION},
                json=payload,
                timeout=90,
            )
            response.raise_for_status()
            data = response.json()
            break
        except requests.ConnectionError as exc:
            if attempt == 2:
                return f"visual_web_search request failed after 3 attempts: {_request_error_detail(exc)}"
            time.sleep(1.0 * (2**attempt))
        except requests.RequestException as exc:
            return f"visual_web_search request failed: {_request_error_detail(exc)}"
        except ValueError as exc:
            return f"visual_web_search returned invalid JSON: {exc}"
    if not isinstance(data, dict):
        return "visual_web_search returned invalid JSON."
    if data.get("error"):
        return f"visual_web_search failed: {data['error']}"
    text = _anthropic_text(data)
    if not text:
        return "visual_web_search returned no text content."
    return text[:8_000]


def _chat_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    content = (choices[0].get("message") or {}).get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [str(item.get("text", "")).strip() for item in content if isinstance(item, dict) and item.get("text")]
        return "\n".join(part for part in parts if part)
    return ""


def deepseek_vlm_search(query: str, image: str = "") -> str:
    q = (query or "").strip()
    if not q:
        return "deepseek_vlm_search failed: query is required."
    if not (image or "").strip():
        return "deepseek_vlm_search failed: no video frame or image has been received yet, so there is no image to analyze."
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return "deepseek_vlm_search failed: configure DeepSeek API key in gateway/tools/config.py."
    payload = {
        "model": "deepseek-v4-flash-vision-exp",
        "thinking": {"type": "disabled"},
        "messages": [
            {"role": "system", "content": _dated_prompt(_VLM_PROMPT)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": q},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}},
                ],
            },
        ],
        "max_tokens": 1024,
    }
    data: Any = None
    for attempt in range(3):
        try:
            response = requests.post(
                _DEEPSEEK_CHAT_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=90,
            )
            response.raise_for_status()
            data = response.json()
            break
        except requests.ConnectionError as exc:
            if attempt == 2:
                return f"deepseek_vlm_search request failed after 3 attempts: {_request_error_detail(exc)}"
            time.sleep(1.0 * (2**attempt))
        except requests.RequestException as exc:
            return f"deepseek_vlm_search request failed: {_request_error_detail(exc)}"
        except ValueError as exc:
            return f"deepseek_vlm_search returned invalid JSON: {exc}"
    if not isinstance(data, dict):
        return "deepseek_vlm_search returned invalid JSON."
    text = _chat_text(data)
    if not text:
        return "deepseek_vlm_search returned no text content."
    return text[:8_000]


_FUNCTIONS: dict[str, Callable[..., str]] = {
    "get_current_datetime": get_current_datetime,
    "DeepSeek": ask_deepseek,
    "deepseek_vlm_search": deepseek_vlm_search,
    "get_current_location": get_current_location,
    "get_city_weather": get_city_weather,
    "visual_web_search": visual_web_search,
}
# Native per-tool override; everything else uses the gateway default.
_TIMEOUTS = {"deepseek_vlm_search": 60.0, "visual_web_search": 60.0}
# Native injects the current frame into these; the model never passes it.
_FRAME_TOOLS = frozenset({"deepseek_vlm_search", "visual_web_search"})


class NativeGatewayTools:
    """Native ``ToolRunner`` semantics behind a ``realtime_tools`` / ``run`` surface."""

    def __init__(self, timeout_s: float = 10.0) -> None:
        self._timeout_s = timeout_s

    @property
    def realtime_tools(self) -> list[dict[str, Any]]:
        return [{"type": "function", **definition["function"]} for definition in _DEFINITIONS]

    async def run(self, name: str, arguments: str, frame_b64: str = "") -> dict[str, str]:
        """Return ``{"status", "output", "error"}``; the model sees ``output or error``."""
        fn = _FUNCTIONS.get(name)
        if fn is None:
            return {"status": "error", "output": "", "error": f"tool not found: {name}"}
        try:
            kwargs = json.loads(arguments or "{}")
            if not isinstance(kwargs, dict):
                raise ValueError("tool arguments must be a JSON object")
            if name in _FRAME_TOOLS:
                kwargs["image"] = frame_b64
            output = await asyncio.wait_for(
                asyncio.to_thread(fn, **kwargs), timeout=_TIMEOUTS.get(name, self._timeout_s)
            )
        except asyncio.TimeoutError:
            return {"status": "timeout", "output": "", "error": "tool timeout"}
        except Exception as exc:
            return {"status": "error", "output": "", "error": str(exc)}
        return {"status": "success", "output": output, "error": ""}
