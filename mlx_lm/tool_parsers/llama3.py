# Copyright © 2026 Apple Inc.

"""Tool calls of Llama 3.1+ (3.1, 3.2, 3.3) with the JSON function-calling template.

The template asks for ``{"name": <function name>, "parameters": <dict>}``. The model writes
that behind the ``<|python_tag|>`` token, or bare when it is the first thing in the answer,
and ends the turn with ``<|eom_id|>`` (a call whose result it waits for) or ``<|eot_id|>``.
Both end tokens are stop tokens in the checkpoint's ``eos_token_id`` and are stripped by the
server; one that is still in the text, as for a checkpoint that does not list ``<|eom_id|>``
as a stop token, ends the call text.

Several calls are written one after another, separated by ``;`` or white space. The text is
a call only if it is made up of calls and nothing else, and every call names a declared
tool: a single call that does not fit (``<|python_tag|>`` followed by code, as in ipython
mode, a built-in tool such as ``brave_search.call(...)``, a name that is not declared,
JSON that is cut off) makes the whole text content, so a call is never run half-way.
``parse_tool_call`` raises :class:`NotAToolCall` for that text, and the server returns the
text as content, without the ``<|python_tag|>`` marker.
"""

import json
from typing import Any

import regex as re

from . import NotAToolCall

tool_call_start = "<|python_tag|>"

# The model ends a call with <|eom_id|> or <|eot_id|>, which stop generation.
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
    # A second marker in one turn separates calls like `;` does.
    body = _drop_terminators(text).replace(tool_call_start, ";")
    names = _declared_names(tools)
    decoder = json.JSONDecoder()
    calls = []
    pos = 0
    while True:
        while pos < len(body) and (body[pos].isspace() or body[pos] == ";"):
            pos += 1
        if pos >= len(body):
            break
        try:
            obj, pos = decoder.raw_decode(body, pos)
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

    The first answer to a request with tools is the bare JSON call, which the text state
    machine does not see as one. The filter sits on the text the state machine emits in the
    normal state. Text that cannot start a call (the first character that is not white space
    is not ``{`` followed by a call key) is released at once. Anything that can is held back
    until the end of the answer, and then returned as a call if all of it is calls to declared
    tools, or released as text otherwise.
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
