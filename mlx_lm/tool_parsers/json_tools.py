# Copyright © 2025 Apple Inc.

import json

tool_call_start = "<tool_call>"

tool_call_end = "</tool_call>"


def _decode_string_arguments(call):
    if isinstance(call, dict) and isinstance(call.get("arguments"), str):
        arguments = json.loads(call["arguments"])
        if not isinstance(arguments, dict):
            raise ValueError("tool call arguments must be a JSON object")
        call["arguments"] = arguments
    return call


def parse_tool_call(text, tools=None):
    parsed = json.loads(text.strip())
    if isinstance(parsed, list):
        return [_decode_string_arguments(call) for call in parsed]
    return _decode_string_arguments(parsed)
