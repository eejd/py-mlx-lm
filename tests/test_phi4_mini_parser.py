# Copyright © 2026 Apple Inc.

import importlib
from types import SimpleNamespace

from mlx_lm.server import ToolCallFormatter
from mlx_lm.tokenizer_utils import _infer_tool_parser

TOOLS = [
    {"type": "function", "function": {"name": "weather"}},
    {"type": "function", "function": {"name": "time"}},
]


def parser():
    return importlib.import_module("mlx_lm.tool_parsers.phi4_mini_json").parse_tool_call


def test_single_and_parallel_calls():
    parse = parser()
    assert parse('{"name":"weather","arguments":{"city":"Paris"}}]  ', TOOLS) == {
        "name": "weather",
        "arguments": {"city": "Paris"},
    }
    assert parse(
        '{"name":"weather","parameters":{}},{"name":"time","arguments":{}}]', TOOLS
    ) == [{"name": "weather", "arguments": {}}, {"name": "time", "arguments": {}}]


def test_json_string_arguments():
    parse = parser()
    assert parse(
        '{"name":"weather","arguments":"{\\"city\\":\\"Paris\\"}"}]', TOOLS
    ) == {
        "name": "weather",
        "arguments": {"city": "Paris"},
    }
    for value in ('"not-json"', '"[]"'):
        import pytest

        with pytest.raises(ValueError):
            parse('{"name":"weather","arguments":' + value + "}]", TOOLS)


def test_formatter_returns_undeclared_name_as_content():
    text = '{"name":"other","arguments":{}}]'
    assert ToolCallFormatter(parser(), TOOLS)([text]) == ([], text)


def test_formatter_drops_invalid_json():
    assert ToolCallFormatter(parser(), TOOLS)(['{"name":']) == ([], "")


def test_template_inference():
    def infer(template, vocab=None):
        return _infer_tool_parser(
            SimpleNamespace(chat_template=template, get_vocab=lambda: vocab or {})
        )

    assert infer("functools[ <tool_call> tool_call.name") == "phi4_mini_json"
    assert infer("[TOOL_CALLS]") == "mistral"
    assert infer("<tool_call>\n<function=") == "qwen3_coder"
    assert infer("Environment: ipython", {"<|python_tag|>": 1}) == "llama3"
