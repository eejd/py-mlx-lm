# Copyright © 2026 Apple Inc.

import json
from typing import Any

from . import NotAToolCall

tool_call_start = "functools["
tool_call_end = ""


def parse_tool_call(text: str, tools: Any | None = None):
    calls = json.loads("[" + text.rstrip())
    if not isinstance(calls, list) or not calls:
        raise ValueError("Expected a nonempty list of tool calls")
    names = {
        function.get("name")
        for tool in tools or []
        if isinstance(tool, dict)
        for function in [tool.get("function", tool)]
        if isinstance(function, dict)
    }
    parsed = []
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            raise ValueError("Invalid tool call")
        arguments = call.get("arguments", call.get("parameters", {}))
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Invalid tool call arguments")
        if call["name"] not in names:
            raise NotAToolCall(text)
        parsed.append({"name": call["name"], "arguments": arguments})
    return parsed[0] if len(parsed) == 1 else parsed
