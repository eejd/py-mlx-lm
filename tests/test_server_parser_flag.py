# Copyright © 2026 Apple Inc.

import argparse
import subprocess
import sys
from unittest import mock

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from mlx_lm.server import ModelProvider
from mlx_lm.tokenizer_utils import load
from mlx_lm.tool_parsers import json_tools


def test_parser_flag_reaches_tokenizer(tmp_path):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")),
        unk_token="<unk>",
        chat_template="{{ messages }}",
    )
    tokenizer.save_pretrained(tmp_path)
    args = argparse.Namespace(
        model="test",
        adapter_path=None,
        draft_model=None,
        pipeline=False,
        trust_remote_code=False,
        chat_template=None,
        tool_call_parser="json_tools",
    )
    with mock.patch("mlx_lm.server.mx.distributed.init") as init:
        init.return_value.size.return_value = 1
        provider = ModelProvider(args)
    result = load(tmp_path, provider._tokenizer_config)
    assert result.init_kwargs["tool_parser_type"] == "json_tools"
    assert result.tool_parser is json_tools.parse_tool_call
    assert load(tmp_path).tool_parser is None


def test_invalid_parser_flag_rejected():
    result = subprocess.run(
        [sys.executable, "-m", "mlx_lm.server", "--tool-call-parser", "not-a-parser"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
