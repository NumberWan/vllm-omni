# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The five tools the native AURA gateway exposes, with its exact schemas.

``_DEFINITIONS`` is the native gateway's ``get_tool_definitions()`` output,
kept verbatim and in order. Looser descriptions or extra search tools make
the model call search for requests it should answer from the frame, such
as drawing a box.
Execution reuses ``AuraToolExecutor`` except ``visual_web_search``.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
from typing import Any

import requests

from examples.online_serving.aura_omni.realtime_web.tools import AuraToolExecutor, _post_bounded_json

_DEFINITIONS: list[dict[str, Any]] = [
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
_DELEGATED = {"get_current_datetime", "DeepSeek", "get_current_location", "get_city_weather"}

_VISUAL_TIMEOUT_SECONDS = 60.0
_VISUAL_MAX_OUTPUT_CHARS = 8_000
_VISUAL_PROMPT = (
    "你是通用图搜助手。你会同时看到当前画面和一条检索问题。"
    "先确认问题指向画面里哪一个清晰可见的对象，再联网检索这个对象需要外部资料才能确定的事实，并据此作答。"
    "用中文简洁准确回答，不要列点，不要 markdown。"
    "看不清、问题和画面对不上、或网上查不到时直接说明。不要编造，也不要只描写外观。"
)


def _error(code: str) -> str:
    return json.dumps({"ok": False, "error": {"code": code}}, separators=(",", ":"))


def _visual_web_search(query: str, frame_b64: str) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return _error("deepseek_api_key_missing")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/anthropic").rstrip("/")
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    body = {
        "model": os.environ.get("DEEPSEEK_VISION_MODEL", "deepseek-flash"),
        "max_tokens": 1024,
        "thinking": {"type": "disabled"},
        "system": f"当前日期时间：{now.strftime('%Y-%m-%d %H:%M')}（北京时间）\n\n{_VISUAL_PROMPT}",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": frame_b64}},
                ],
            }
        ],
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
    }
    try:
        payload = _post_bounded_json(
            f"{base_url}/v1/messages",
            json_body=body,
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            timeout_seconds=_VISUAL_TIMEOUT_SECONDS,
        )
    except (requests.RequestException, ValueError):
        return _error("visual_web_search_upstream_failed")
    parts = [
        str(block.get("text") or "").strip()
        for block in payload.get("content") or []
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    text = "\n".join(part for part in parts if part)[:_VISUAL_MAX_OUTPUT_CHARS]
    if not text:
        return _error("visual_web_search_empty_response")
    return json.dumps({"ok": True, "result": {"query": query, "summary": text}}, ensure_ascii=False)


class NativeGatewayTools:
    """Same ``realtime_tools`` / ``run`` surface as ``AuraToolExecutor``."""

    def __init__(self) -> None:
        self._executor = AuraToolExecutor()

    @property
    def realtime_tools(self) -> list[dict[str, Any]]:
        return [{"type": "function", **definition["function"]} for definition in _DEFINITIONS]

    async def run(self, name: str, arguments: str, frame_b64: str = "") -> str:
        if name in _DELEGATED:
            if name == "get_city_weather":
                try:
                    parsed = json.loads(arguments or "{}")
                except ValueError:
                    return _error("invalid_tool_arguments")
                if isinstance(parsed, dict) and "date" in parsed:
                    if str(parsed.pop("date") or "").strip():
                        return _error("weather_date_unsupported")
                    arguments = json.dumps(parsed, ensure_ascii=False)
            return await self._executor.run(name, arguments)
        if name == "visual_web_search":
            try:
                query = str(json.loads(arguments or "{}").get("query") or "").strip()
            except (ValueError, AttributeError):
                return _error("invalid_tool_arguments")
            if not query:
                return _error("invalid_tool_arguments")
            if not frame_b64:
                return _error("no_frame")
            return await asyncio.to_thread(_visual_web_search, query, frame_b64)
        return _error("unknown_tool")
