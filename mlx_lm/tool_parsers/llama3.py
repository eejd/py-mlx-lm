# Copyright © 2026 Apple Inc.

"""Tool calls of Llama 3.1+ (3.1, 3.2, 3.3) with the JSON function-calling template.

The model writes ``{"name": <function name>, "parameters": <dict>}`` behind the
``<|python_tag|>`` token, or bare as the first thing in the answer. Several calls follow
each other, separated by ``;`` or white space. The end tokens ``<|eom_id|>`` and
``<|eot_id|>`` stop generation, so the server removes them. One that is still in the text
ends the call text.

The text is a call only if all of it is calls to declared tools. Anything else (code behind
the tag, a built-in tool, an undeclared name, cut-off JSON) raises ``NotAToolCall``, and the
server returns the text as content without the tag, so a call is never run half-way.
"""

import json
from typing import Any

import regex as re

from . import NotAToolCall

tool_call_start = "<|python_tag|>"

# A call ends with <|eom_id|> or <|eot_id|>, which stop generation.
tool_call_end = ""

_terminators = ("<|eom_id|>", "<|eot_id|>", "<|end_of_text|>")

# `type` is "function" when present. `id` is passed on to the server.
_call_keys = {"type", "name", "parameters", "arguments", "id"}

# What a bare call starts with. The object always opens with one of the call keys.
_bare_call_head = re.compile(r'\{\s*"(?:type|name|parameters|arguments|id)"\s*:')


def _declared_names(tools: Any | None) -> set[str]:
    names = set()
    for tool in tools or []:
        function = tool.get("function", tool) if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _drop_terminators(text: str) -> str:
    """The text up to the first end-of-turn token."""
    ends = [i for t in _terminators if (i := text.find(t)) >= 0]
    return text[: min(ends)] if ends else text


def _to_call(obj: Any, names: set[str]) -> dict[str, Any] | None:
    if not isinstance(obj, dict) or not obj.keys() <= _call_keys:
        return None
    name = obj.get("name")
    if not isinstance(name, str) or name not in names:
        return None
    if obj.get("type", "function") != "function":
        return None
    if "parameters" in obj and "arguments" in obj:
        return None
    arguments = obj.get("parameters", obj.get("arguments", {}))
    if isinstance(arguments, str):
        # A template that renders the arguments as a JSON string gets them back that way.
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    call = dict(name=name, arguments=arguments)
    if isinstance(obj.get("id"), str):
        call["id"] = obj["id"]
    return call


def parse_calls(text: str, tools: Any | None = None) -> list[dict[str, Any]]:
    """Every call in ``text``; raises :class:`NotAToolCall` unless all of it is calls."""
    content = _drop_terminators(text).replace(tool_call_start, "")
    names = _declared_names(tools)
    decoder = json.JSONDecoder()
    calls = []
    pos = 0
    while True:
        # Between calls: white space, ``;`` and the marker, which also separates calls.
        while pos < len(text):
            if text[pos].isspace() or text[pos] == ";":
                pos += 1
            elif text.startswith(tool_call_start, pos):
                pos += len(tool_call_start)
            else:
                break
        if pos >= len(text) or text.startswith(_terminators, pos):
            break
        try:
            obj, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            raise NotAToolCall(content) from None
        call = _to_call(obj, names)
        if call is None:
            raise NotAToolCall(content)
        calls.append(call)
    if not calls:
        raise NotAToolCall(content)
    return calls


def parse_tool_call(text: str, tools: Any | None = None):
    calls = parse_calls(text, tools)
    return calls[0] if len(calls) == 1 else calls


class BareToolCallFilter:
    """Recovers a call the model wrote without ``<|python_tag|>``.

    It sits on the text the state machine emits in the normal state. Text that cannot start
    a call (not ``{`` and a call key after white space) is released at once. Other text is
    held until the end of the answer, then returned as a call if all of it is calls to
    declared tools, or released as text.
    """

    def __init__(self, tools: Any | None = None):
        self._tools = tools
        self._buf = ""
        self._passing = False

    def feed(self, text: str) -> tuple[str, list[str]]:
        if self._passing:
            return text, []
        self._buf += text
        head = self._buf.lstrip()
        if not head:
            return "", []
        if head[0] == "{" and _bare_call_head.match(head, partial=True):
            return "", []
        out, self._buf, self._passing = self._buf, "", True
        return out, []

    def flush(self) -> tuple[str, list[str]]:
        held, self._buf, self._passing = self._buf, "", False
        if not held:
            return "", []
        try:
            parse_calls(held, self._tools)
        except NotAToolCall:
            return held, []
        return "", [held]


def bare_tool_call_filter(tools: Any | None = None) -> BareToolCallFilter:
    """The server asks the parser module for a filter of calls without the start marker."""
    return BareToolCallFilter(tools)
