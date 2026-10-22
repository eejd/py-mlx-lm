# Copyright © 2024 Apple Inc.

import contextlib
import http
import io
import json
import threading
import time
import unittest
from queue import Queue
from unittest import mock

import mlx.core as mx
import requests

from mlx_lm.generate import TextStateMachine
from mlx_lm.models.cache import KVCache, QuantizedKVCache
from mlx_lm.server import (
    APIHandler,
    BareToolCallFilter,
    GenerationContext,
    LRUPromptCache,
    Response,
    ResponseGenerator,
    SamplingArguments,
    ToolCallFormatter,
    _make_sampler,
)
from mlx_lm.tool_parsers import llama3, pythonic, qwen3_coder
from mlx_lm.utils import load


class DummyModelProvider:
    def __init__(self, with_draft=False, kv_bits=None, quantized_kv_start=0):
        HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        self.model, self.tokenizer = load(HF_MODEL_PATH)
        self.processor = None
        self.model_key = (HF_MODEL_PATH, None)
        self.is_batchable = True

        # Add draft model support
        self.draft_model = None
        self.draft_model_key = None
        self.cli_args = type(
            "obj",
            (object,),
            {
                "adapter_path": None,
                "chat_template": None,
                "use_default_chat_template": False,
                "trust_remote_code": False,
                "draft_model": None,
                "num_draft_tokens": 3,
                "temp": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "max_tokens": 512,
                "chat_template_args": {},
                "model": None,
                "decode_concurrency": 32,
                "prompt_concurrency": 8,
                "prefill_step_size": 2048,
                "prompt_cache_size": 10,
                "prompt_cache_bytes": 1 << 63,
                "prompt_cache_total_bytes": None,
                "allowed_origins": ["*"],
                "kv_bits": kv_bits,
                "kv_group_size": 64,
                "quantized_kv_start": quantized_kv_start,
            },
        )

        if with_draft:
            # Use the same model as the draft model for testing
            self.draft_model, _ = load(HF_MODEL_PATH)
            self.draft_model_key = HF_MODEL_PATH
            self.cli_args.draft_model = HF_MODEL_PATH

    def load(self, model, adapter=None, draft_model=None):
        assert model in ["default_model", "chat_model"]
        return self.model, self.tokenizer

    def load_default(self):
        return self.load("default_model", None, "default_model")

    def reset(self) -> None:
        self.model_key = None
        self.model = None
        self.tokenizer = None
        self.draft_model = None
        self.is_batchable = False


class MockCache:
    def __init__(self, value, is_trimmable: bool = True):
        self.value = value
        self._is_trimmable = is_trimmable

    @property
    def nbytes(self):
        return len(self.value)

    def __eq__(self, other):
        return other.value == self.value

    def is_trimmable(self):
        return self._is_trimmable

    def trim(self, n):
        assert self._is_trimmable
        return n


class TestBareToolCallFilter(unittest.TestCase):
    """A model that skips the <tool_call> marker still gets its call parsed."""

    CALL = "<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>"
    TOOLS = [
        {"type": "function", "function": {"name": "get_weather"}},
        {"type": "function", "function": {"name": "read_file"}},
        {"type": "function", "function": {"name": "write_file"}},
    ]

    def make(self, tools=TOOLS):
        return BareToolCallFilter.for_parser(qwen3_coder.parse_tool_call, tools)

    def run_chunks(self, chunks, tools=TOOLS):
        f = self.make(tools)
        text, calls = "", []
        for chunk in chunks:
            t, c = f.feed(chunk)
            text += t
            calls += c
        rest, rest_calls = f.flush()
        return text + rest, calls + rest_calls

    def drive(self, chunks):
        """The server's generation loop in miniature: state machine, then the filter."""
        sm = TextStateMachine(
            {"normal": [("<tool_call>", "tool")], "tool": [("</tool_call>", "normal")]}
        )
        state, prev = sm.make_state(), "normal"
        f = self.make()
        text, tool_text, calls = "", "", []
        for chunk in chunks:
            state, clean, current = sm.step(state, chunk)
            if prev == "normal" and current != "normal":
                held, held_calls = f.flush()
                text += held
                calls += held_calls
            if current == "tool":
                tool_text += clean
            elif current == "normal":
                if prev == "tool":
                    calls.append(tool_text)
                    tool_text = ""
                clean, bare = f.feed(clean)
                calls += bare
                text += clean
            prev = current
        state, rest, _ = sm.flush(state)
        t, bare = f.feed(rest)
        held, held_calls = f.flush()
        return text + t + held, calls + bare + held_calls

    def test_only_parsers_that_declare_markers_get_a_filter(self):
        self.assertIsNotNone(self.make())
        self.assertIsNone(BareToolCallFilter.for_parser(pythonic.parse_tool_call, self.TOOLS))
        self.assertIsNone(BareToolCallFilter.for_parser(None, self.TOOLS))

    def test_tool_names(self):
        names = BareToolCallFilter.tool_names(self.TOOLS + [{"name": "plain"}, None, "x", {}])
        self.assertEqual(names, {"get_weather", "read_file", "write_file", "plain"})

    def test_bare_call_with_trailer(self):
        text, calls = self.run_chunks([self.CALL + "\n</tool_call>"])
        self.assertEqual((text, calls), ("", [self.CALL]))

    def test_bare_call_without_trailer(self):
        text, calls = self.run_chunks([self.CALL])
        self.assertEqual((text, calls), ("", [self.CALL]))

    def test_trailing_whitespace_after_a_call_is_not_content(self):
        for tail in ("\n", "\n</tool_call>\n", "  \n</tool_call>"):
            text, calls = self.run_chunks([self.CALL + tail])
            self.assertEqual((text, calls), ("", [self.CALL]), repr(tail))

    def test_every_chunking_gives_the_same_result(self):
        full = "Sure.\n" + self.CALL + "\n</tool_call>"
        for size in (1, 2, 3, 5, 7, len(full)):
            chunks = [full[i : i + size] for i in range(0, len(full), size)]
            text, calls = self.run_chunks(chunks)
            self.assertEqual((text, calls), ("Sure.\n", [self.CALL]), size)

    def test_parallel_bare_calls(self):
        other = self.CALL.replace("get_weather", "read_file")
        text, calls = self.run_chunks([self.CALL + "\n</tool_call>\n" + other + "\n</tool_call>"])
        self.assertEqual((text, calls), ("", [self.CALL, other]))

    def test_text_after_the_call_is_kept(self):
        text, calls = self.run_chunks([self.CALL + "\n</tool_call>", " Done."])
        self.assertEqual((text, calls), (" Done.", [self.CALL]))

    def test_a_non_trailer_after_the_call_is_released(self):
        text, calls = self.run_chunks([self.CALL + "\n</oops>"])
        self.assertEqual((text, calls), ("\n</oops>", [self.CALL]))

    def test_partial_marker_that_is_not_a_call_is_released(self):
        text, calls = self.run_chunks(["a <func", "tion of x"])
        self.assertEqual((text, calls), ("a <function of x", []))

    def test_text_ending_in_a_partial_marker_is_released_on_flush(self):
        text, calls = self.run_chunks(["call it <fun"])
        self.assertEqual((text, calls), ("call it <fun", []))

    def test_unfinished_call_is_released_as_text(self):
        cut = "<function=get_weather>\n<parameter=city>\nTok"
        text, calls = self.run_chunks([cut])
        self.assertEqual((text, calls), (cut, []))

    def test_undeclared_name_is_prose_and_does_not_stall(self):
        f = self.make()
        out, calls = f.feed("Qwen uses <function=name> format. ")
        self.assertEqual((out, calls), ("Qwen uses <function=name> format. ", []))
        out, _ = f.feed("More text")
        self.assertEqual(out, "More text")

    def test_marker_not_followed_by_a_name_is_prose(self):
        text, calls = self.run_chunks(["see <function= here and <function=a b> too"])
        self.assertEqual((text, calls), ("see <function= here and <function=a b> too", []))

    def test_end_marker_inside_a_value_does_not_cut_the_call(self):
        call = (
            "<function=write_file>\n<parameter=content>\nx = '</function>'\n</parameter>\n"
            "<parameter=path>\n/a.py\n</parameter>\n</function>"
        )
        text, calls = self.run_chunks([call + "\n</tool_call>"])
        self.assertEqual((text, calls), ("", [call]))

    def test_recovered_call_parses_to_a_tool_call(self):
        _, calls = self.run_chunks([self.CALL + "\n</tool_call>"])
        out, _ = ToolCallFormatter(qwen3_coder.parse_tool_call, tools=None)(calls)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(out[0]["function"]["arguments"]), {"city": "Tokyo"})

    def test_wrapped_and_bare_calls_mix_in_order(self):
        wrapped = self.CALL.replace("get_weather", "read_file")
        full = "Hi. <tool_call>\n" + wrapped + "\n</tool_call>\n" + self.CALL + "\n</tool_call>"
        for size in (1, 4, len(full)):
            chunks = [full[i : i + size] for i in range(0, len(full), size)]
            text, calls = self.drive(chunks)
            # Whitespace between a wrapped call and what follows is existing behaviour.
            self.assertEqual(text.strip(), "Hi.", size)
            self.assertEqual(len(calls), 2, size)
            self.assertIn("read_file", calls[0])
            self.assertIn("get_weather", calls[1])

    def test_flush_releases_a_held_marker_before_another_state(self):
        f = self.make()
        out, _ = f.feed("see <fun")
        self.assertEqual(out, "see ")
        self.assertEqual(f.flush(), ("<fun", []))

    def test_state_change_in_the_loop_releases_the_held_marker_in_order(self):
        text, calls = self.drive(["see <fun", "<tool_call>\n" + self.CALL + "\n</tool_call>"])
        self.assertEqual(text.rstrip(), "see <fun")
        self.assertEqual(len(calls), 1)

    def test_a_call_missing_a_parameter_closer_is_recovered_at_the_end(self):
        sloppy = (
            "<function=get_weather>\n<parameter=city>\nTokyo\n<parameter=days>\n3\n"
            "</function>\n</tool_call>"
        )
        text, calls = self.run_chunks([sloppy])
        self.assertEqual(text, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].count("<parameter="), calls[0].count("</parameter>"))
        out, _ = ToolCallFormatter(qwen3_coder.parse_tool_call, tools=None)(calls)
        self.assertEqual(
            json.loads(out[0]["function"]["arguments"]), {"city": "Tokyo", "days": "3"}
        )

    def test_parallel_calls_with_a_sloppy_first_are_not_merged(self):
        sloppy = "<function=get_weather>\n<parameter=city>\nTokyo\n</function>"
        good = "<function=read_file>\n<parameter=path>\n/a.py\n</parameter>\n</function>"
        text, calls = self.run_chunks([sloppy + "\n</tool_call>\n" + good + "\n</tool_call>"])
        self.assertEqual(text, "")
        self.assertEqual(len(calls), 2)
        out, _ = ToolCallFormatter(qwen3_coder.parse_tool_call, tools=None)(calls)
        self.assertEqual(
            [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in out],
            [("get_weather", {"city": "Tokyo"}), ("read_file", {"path": "/a.py"})],
        )

    def test_two_sloppy_calls_and_trailing_text(self):
        one = "<function=get_weather>\n<parameter=city>\nTokyo\n</function>"
        two = "<function=read_file>\n<parameter=path>\n/a.py\n</function>"
        text, calls = self.run_chunks([one + "\n</tool_call>\n" + two + "\n</tool_call>\nbye"])
        self.assertEqual((len(calls), text), (2, "\nbye"))

    def test_a_cut_off_call_stays_text_even_with_a_name(self):
        cut = "<function=get_weather>\n<parameter=city>\nTokyo\n<parameter=days>\n3"
        text, calls = self.run_chunks([cut])
        self.assertEqual((text, calls), (cut, []))

    def test_no_declared_tools_means_nothing_is_recovered(self):
        text, calls = self.run_chunks([self.CALL + "\n</tool_call>"], tools=[])
        self.assertEqual(calls, [])


class ScriptedHandler:
    """Real HTTP requests through APIHandler.handle_completion with a scripted token stream.

    Only the generation thread is replaced: ``ResponseGenerator.generate`` returns the
    text chunks a model would produce, with the real tool parser (``PARSER``) and the real
    text state machine (``TRANSITIONS``) in the handler's loop.
    """

    PARSER = staticmethod(qwen3_coder.parse_tool_call)
    TRANSITIONS = {
        "normal": [("<tool_call>", "tool")],
        "tool": [("</tool_call>", "normal")],
    }
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]

    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(
            DummyModelProvider(), LRUPromptCache()
        )
        cls.httpd = http.server.HTTPServer(
            ("localhost", 0),
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()

    def scripted(self, chunks):
        def generate(request, args, progress_callback=None):
            ctx = GenerationContext(
                has_tool_calling=True,
                has_thinking=False,
                tool_parser=self.PARSER,
                text_sm=TextStateMachine(self.TRANSITIONS),
                initial_state="normal",
                prompt=[1, 2, 3],
            )

            def responses():
                for chunk in chunks:
                    yield Response(chunk, 7, 0.0, None, ())
                # the stop token carries no text; the handler discards the buffer on it
                yield Response("", 2, 0.0, "stop", ())

            return ctx, responses()

        return mock.patch.object(self.response_generator, "generate", generate)

    def post(self, chunks, tools=TOOLS, stream=False):
        body = {
            "model": "chat_model",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "weather in Tokyo?"}],
            "stream": stream,
        }
        if tools is not None:
            body["tools"] = tools
        with self.scripted(chunks):
            r = requests.post(
                f"http://localhost:{self.port}/v1/chat/completions",
                json=body,
                stream=stream,
            )
            self.assertEqual(r.status_code, 200, r.text)
            if not stream:
                choice = r.json()["choices"][0]
                return choice["message"], choice["finish_reason"]
            content, calls, finish = "", [], None
            for line in r.iter_lines():
                if not line.startswith(b"data: ") or line.endswith(b"[DONE]"):
                    continue
                choice = json.loads(line[6:])["choices"][0]
                delta = choice.get("delta", {})
                content += delta.get("content") or ""
                calls += delta.get("tool_calls") or []
                finish = choice.get("finish_reason") or finish
            return {"content": content, "tool_calls": calls}, finish



class TestBareToolCallsThroughTheHandler(ScriptedHandler, unittest.TestCase):
    CALL = "<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>"

    def check_call(self, message, finish):
        self.assertEqual(finish, "tool_calls")
        calls = message["tool_calls"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Tokyo"})

    def test_bare_call_is_recovered(self):
        message, finish = self.post(["<function=get_weather>\n", "<parameter=city>\nTokyo\n</parameter>\n</function>", "\n</tool_call>"])
        self.check_call(message, finish)
        self.assertFalse((message["content"] or "").strip())

    def test_bare_call_is_recovered_when_streaming(self):
        message, finish = self.post(
            ["<function=get_weather>\n", "<parameter=city>\nTokyo\n</parameter>\n</function>", "\n</tool_call>"],
            stream=True,
        )
        self.check_call(message, finish)

    def test_plain_text_without_a_call_is_unchanged(self):
        message, finish = self.post(["It is ", "sunny in ", "Tokyo."])
        self.assertEqual(finish, "stop")
        self.assertEqual(message["content"], "It is sunny in Tokyo.")
        self.assertFalse(message.get("tool_calls"))

    def test_held_marker_prefix_is_released_in_order_before_a_wrapped_call(self):
        # "<fun" is held as a possible bare call; the wrapped call that follows must not
        # swallow it or reorder it.
        message, finish = self.post(
            ["see <fun", "<tool_call>\n" + self.CALL + "\n</tool_call>"]
        )
        self.check_call(message, finish)
        self.assertEqual(message["content"].strip(), "see <fun")

    def test_held_text_is_released_at_a_state_change_inside_one_chunk(self):
        # "<fun" is held as a possible bare call when the same chunk goes on into a wrapped
        # call. If it is not released at that transition it stays in the filter's buffer
        # and fuses with the text after the call ("<fun" + "ction=...>" would then be
        # recovered as a second, bogus call).
        after = (
            "ction=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>"
        )
        message, finish = self.post(
            ["see <fun<tool_call>\n" + self.CALL + "\n</tool_call>" + after]
        )
        self.check_call(message, finish)
        self.assertEqual(message["content"].strip(), "see <fun" + after)

    def test_without_tools_a_bare_block_stays_text(self):
        message, finish = self.post(["<function=get_weather>\n</function>"], tools=None)
        self.assertEqual(finish, "stop")
        self.assertIn("<function=get_weather>", message["content"])


class TestLlama3ToolCallsThroughTheHandler(ScriptedHandler, unittest.TestCase):
    """Llama 3.x writes ``<|python_tag|>{json}`` or, as its first answer, the bare JSON."""

    PARSER = staticmethod(llama3.parse_tool_call)
    TRANSITIONS = {"normal": [("<|python_tag|>", "tool")], "tool": []}
    CALL = '{"name": "get_weather", "parameters": {"city": "Tokyo"}}'

    def check_call(self, message, finish, city="Tokyo"):
        self.assertEqual(finish, "tool_calls")
        calls = message["tool_calls"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": city})
        self.assertFalse((message["content"] or "").strip())

    def check_content(self, message, finish, content):
        self.assertEqual(finish, "stop")
        self.assertEqual(message["content"], content)
        self.assertFalse(message.get("tool_calls"))

    def chunked(self, text, size):
        return [text[i : i + size] for i in range(0, len(text), size)]

    def test_tagged_call(self):
        for stream in (False, True):
            for size in (1, 5, 1000):
                message, finish = self.post(
                    self.chunked("<|python_tag|>" + self.CALL, size), stream=stream
                )
                self.check_call(message, finish)
                self.assertNotIn("python_tag", message["content"] or "")

    def test_bare_call(self):
        for stream in (False, True):
            for size in (1, 5, 1000):
                message, finish = self.post(self.chunked(self.CALL, size), stream=stream)
                self.check_call(message, finish)

    def test_bare_call_with_the_type_key(self):
        call = '{"type": "function", "name": "get_weather", "parameters": {"city": "Oslo"}}'
        for stream in (False, True):
            message, finish = self.post(self.chunked(call, 3), stream=stream)
            self.check_call(message, finish, city="Oslo")

    def test_semicolon_separated_calls_are_all_returned(self):
        other = self.CALL.replace("Tokyo", "Paris")
        for text in ("<|python_tag|>" + self.CALL + "; " + other, self.CALL + ";" + other):
            for stream in (False, True):
                message, finish = self.post(self.chunked(text, 7), stream=stream)
                self.assertEqual(finish, "tool_calls")
                self.assertEqual(
                    [json.loads(c["function"]["arguments"])["city"] for c in message["tool_calls"]],
                    ["Tokyo", "Paris"],
                )
                if stream:
                    self.assertEqual([c["index"] for c in message["tool_calls"]], [0, 1])

    def test_text_before_the_tag_is_kept(self):
        message, finish = self.post(["Let me check. ", "<|python_tag|>", self.CALL])
        self.assertEqual(finish, "tool_calls")
        self.assertEqual(message["content"], "Let me check. ")
        self.assertEqual(len(message["tool_calls"]), 1)

    def test_code_after_the_tag_is_content_without_the_tag(self):
        text = "import json\nprint(json.dumps({'a': 1}))"
        for stream in (False, True):
            message, finish = self.post(["<|python_tag|>", text], stream=stream)
            self.check_content(message, finish, text)

    def test_undeclared_name_is_content(self):
        call = self.CALL.replace("get_weather", "get_time")
        for stream in (False, True):
            message, finish = self.post(["<|python_tag|>", call], stream=stream)
            self.check_content(message, finish, call)
            message, finish = self.post([call], stream=stream)
            self.check_content(message, finish, call)

    def test_malformed_json_is_content(self):
        cut = self.CALL[:-8]
        for stream in (False, True):
            message, finish = self.post(["<|python_tag|>", cut], stream=stream)
            self.check_content(message, finish, cut)
            message, finish = self.post(self.chunked(cut, 4), stream=stream)
            self.check_content(message, finish, cut)

    def test_a_call_with_trailing_prose_is_not_run(self):
        text = self.CALL + " That is the weather call."
        message, finish = self.post(self.chunked(text, 6))
        self.check_content(message, finish, text)

    def test_json_answer_that_is_not_a_call_is_content(self):
        for text in ('{"city": "Tokyo", "temp": 21}', '{"name": "Tokyo", "temp": 21}'):
            for stream in (False, True):
                message, finish = self.post(self.chunked(text, 3), stream=stream)
                self.check_content(message, finish, text)

    def test_plain_text_is_unchanged(self):
        message, finish = self.post(["It is ", "sunny in ", "Tokyo."])
        self.check_content(message, finish, "It is sunny in Tokyo.")

    def test_without_tools_the_tag_text_is_content(self):
        message, finish = self.post(["<|python_tag|>", self.CALL], tools=None)
        self.check_content(message, finish, self.CALL)

    def test_end_of_turn_token_in_the_text_ends_the_call(self):
        message, finish = self.post(["<|python_tag|>", self.CALL + "<|eom_id|>"])
        self.check_call(message, finish)

    def test_html_entities_in_arguments_are_kept(self):
        call = '{"name": "get_weather", "parameters": {"city": "A &amp; B &lt;1&gt;"}}'
        message, finish = self.post(["<|python_tag|>", call])
        self.check_call(message, finish, city="A &amp; B &lt;1&gt;")


class TestTextStateMachine(unittest.TestCase):
    """Test the TextStateMachine buffering and stripping behavior."""

    def test_strips_control_sequences(self):
        sm = TextStateMachine(
            {
                "normal": [("<tool_call>", "tool")],
                "tool": [("</tool_call>", "normal")],
            }
        )
        state = sm.make_state()
        state, text, _ = sm.step(state, "hi <tool_call>body</tool_call> bye")
        state, rest, _ = sm.flush(state)
        full = text + rest
        self.assertEqual(full, "hi body bye")

    def test_back_to_back_tool_calls(self):
        sm = TextStateMachine(
            {
                "normal": [("<tool_call>", "tool")],
                "tool": [("</tool_call>", "normal")],
            }
        )
        state = sm.make_state()
        state, t1, _ = sm.step(state, "<tool_call>call1</tool_call>")
        state, t2, _ = sm.step(state, "<tool_call>call2</tool_call>")
        state, rest, _ = sm.flush(state)
        full = t1 + t2 + rest
        self.assertEqual(full, "call1call2")

    def test_partial_match_buffered_then_flushed(self):
        sm = TextStateMachine(
            {
                "normal": [("<tool_call>", "tool")],
                "tool": [("</tool_call>", "normal")],
            }
        )
        # First enter tool state
        state = sm.make_state()
        state, text, s = sm.step(state, "<tool_call>body</")
        self.assertEqual(s, "tool")
        # 'body' is emitted, '</' is buffered (partial match of '</tool_call>')
        self.assertEqual(text, "body")
        # flush releases the buffered text
        state, rest, s = sm.flush(state)
        self.assertEqual(rest, "</")

    def test_discard_drops_buffer(self):
        sm = TextStateMachine(
            {
                "normal": [("STOP", "normal")],
            }
        )
        state = sm.make_state()
        state, text, s = sm.step(state, "hello ST")
        self.assertEqual(text, "hello ")
        # discard drops the buffered 'ST'
        state, s = sm.discard(state)
        self.assertEqual(s, "normal")

    def test_stop_words_stripped(self):
        sm = TextStateMachine(
            {
                "normal": [("STOP", "normal")],
            }
        )
        state = sm.make_state()
        state, text, _ = sm.step(state, "hello STOP world")
        state, rest, _ = sm.flush(state)
        self.assertEqual(text + rest, "hello  world")

    def test_reasoning_to_tool_transition(self):
        # A tool call started inside a reasoning block must enter "tool".
        sm = TextStateMachine(
            {
                "normal": [("<think>", "reasoning"), ("<tool>", "tool")],
                "reasoning": [("</think>", "normal"), ("<tool>", "tool")],
                "tool": [("</tool>", "normal")],
            }
        )
        state = sm.make_state()
        state, _, s = sm.step(state, "<think>hmm")
        self.assertEqual(s, "reasoning")
        state, _, s = sm.step(state, "<tool>")
        self.assertEqual(s, "tool")
        state, _, s = sm.step(state, "</tool>")
        self.assertEqual(s, "normal")

    def test_empty_end_marker_stays_in_tool_on_discard(self):
        # Models with an empty tool_call_end (e.g. Mistral) never leave "tool";
        # discard on stop must preserve the state so the tool call is flushed.
        sm = TextStateMachine(
            {
                "normal": [("[TOOL_CALLS]", "tool")],
                "tool": [],
            }
        )
        state = sm.make_state()
        state, text, s = sm.step(state, "[TOOL_CALLS]f[ARGS]{}")
        self.assertEqual(s, "tool")
        self.assertEqual(text, "f[ARGS]{}")
        state, s = sm.discard(state)
        self.assertEqual(s, "tool")

    def test_split_keeps_text_in_its_state(self):
        # Held text released by a transition belongs to the state before it.
        sm = TextStateMachine(
            {
                "normal": [("<think>", "reasoning"), ("<tool_call>", "tool")],
                "reasoning": [("</think>", "normal")],
                "tool": [("</tool_call>", "normal")],
            }
        )
        state = sm.make_state()
        state, segments = sm.split(state, "x <")
        self.assertEqual(segments, [("normal", "x ")])
        state, segments = sm.split(state, "<tool_call>{}")
        self.assertEqual(segments, [("normal", "<"), ("tool", "{}")])
        state, segments = sm.split(state, "</tool_call>")
        self.assertEqual(segments, [("tool", ""), ("normal", "")])

    def test_held_text_read_once(self):
        sm = TextStateMachine({"normal": [("##", "normal")]})
        state = sm.make_state()
        state, t1, _ = sm.step(state, "a #")
        state, t2, _ = sm.step(state, "# b")
        state, rest, _ = sm.flush(state)
        self.assertEqual(t1 + t2 + rest, "a  b")


class TestToolCallFormatter(unittest.TestCase):
    def test_formats_parallel_tool_calls(self):
        formatter = ToolCallFormatter(pythonic.parse_tool_call, tools=None)
        raw_tool_call = (
            '[get_time(location="Paris"), '
            'grocery.order(items=[{"name": "noodles", "organic": true}])]'
        )

        tool_calls, content = formatter([raw_tool_call])

        self.assertEqual(
            [tc["function"]["name"] for tc in tool_calls],
            ["get_time", "grocery.order"],
        )
        self.assertTrue(all(tc["type"] == "function" for tc in tool_calls))
        self.assertEqual(
            json.loads(tool_calls[1]["function"]["arguments"]),
            {"items": [{"name": "noodles", "organic": True}]},
        )
        self.assertEqual(content, "")
        self.assertTrue(formatter.made_call)

    def test_text_that_is_not_a_call_is_returned_as_content(self):
        tools = [{"type": "function", "function": {"name": "get_time"}}]
        formatter = ToolCallFormatter(llama3.parse_tool_call, tools)

        tool_calls, content = formatter(["print(1)"])

        self.assertEqual((tool_calls, content), ([], "print(1)"))
        self.assertFalse(formatter.made_call)

        tool_calls, content = formatter(['{"name": "get_time", "parameters": {}}', "x = 1"])

        self.assertEqual([tc["function"]["name"] for tc in tool_calls], ["get_time"])
        self.assertEqual(content, "x = 1")
        self.assertTrue(formatter.made_call)

    def test_a_call_cut_short_is_dropped(self):
        formatter = ToolCallFormatter(qwen3_coder.parse_tool_call, tools=None)

        self.assertEqual(formatter(["<function=a>\n<parameter=b>"]), ([], ""))
        self.assertTrue(formatter.made_call)


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(
            DummyModelProvider(), LRUPromptCache()
        )
        cls.server_address = ("localhost", 0)
        cls.httpd = http.server.HTTPServer(
            cls.server_address,
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()

    def test_max_tokens_zero_is_rejected(self):
        url = f"http://localhost:{self.port}/v1/completions"
        response = requests.post(url, json={"prompt": "hi", "max_tokens": 0})
        self.assertEqual(response.status_code, 400)

    def test_handle_completions(self):
        url = f"http://localhost:{self.port}/v1/completions"

        post_data = {
            "model": "default_model",
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 0.5,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "repetition_context_size": 20,
            "seed": 999,
            "stop": "stop sequence",
        }

        response = requests.post(url, json=post_data)

        response_body = json.loads(response.text)

        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        first_text = response_body["choices"][0]["text"]
        self.assertEqual(
            first_text,
            json.loads(requests.post(url, json=post_data).text)["choices"][0]["text"],
        )

    def test_handle_chat_completions(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_with_content_fragments(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant."}
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_with_null_tool_content(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {"role": "user", "content": "what is 2+3?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "123",
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 2, "b": 3}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "5", "tool_call_id": "123"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_with_dict_tool_arguments(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "what is 2+3?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "123",
                            "function": {
                                "name": "add",
                                "arguments": {"a": 2, "b": 3},
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "5", "tool_call_id": "123"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_generation_thread_exit_releases_model(self):
        response_generator = ResponseGenerator(DummyModelProvider(), LRUPromptCache())
        provider = response_generator.model_provider
        response_generator.stop_and_join()

        # The weights are dropped even though this frame still holds the
        # provider, and the args survive for request handler threads.
        self.assertIsNone(provider.model)
        self.assertIsNone(provider.tokenizer)
        self.assertFalse(provider.is_batchable)
        self.assertIsNotNone(response_generator.cli_args.allowed_origins)

    def test_make_state_machine_empty_tool_call_end(self):
        class FakeTokenizer:
            has_thinking = False
            has_tool_calling = True
            tool_call_start = "[TOOL_CALLS]"
            tool_call_end = ""
            tool_call_start_tokens = (100,)
            tool_call_end_tokens = ()
            eos_token_ids = [2]
            structural_markers = ()
            response_transitions = {}

            def convert_ids_to_tokens(self, t):
                return f"<eos{t}>"

            def encode(self, text, add_special_tokens=False):
                return []

        stop_sequences, text_sm = self.response_generator._make_state_machine(
            ("fake-empty-end", None, None),
            FakeTokenizer(),
            stop_words=[],
        )

        # Verify the text state machine strips tool call markers
        text_state = text_sm.make_state()
        text_state, clean_text, s = text_sm.step(text_state, "hello[TOOL_CALLS]body")
        self.assertEqual(s, "tool")
        # 'hello' is before the match, 'body' flows through (no tool_call_end)
        self.assertEqual(clean_text, "hellobody")

        # Verify EOS stops via the stop matcher
        self.assertTrue(stop_sequences.matcher().advance(2))

    def test_handle_models(self):
        url = f"http://localhost:{self.port}/v1/models"
        response = requests.get(url)
        self.assertEqual(response.status_code, 200)
        response_body = json.loads(response.text)
        self.assertEqual(response_body["object"], "list")
        self.assertIsInstance(response_body["data"], list)
        self.assertGreater(len(response_body["data"]), 0)
        model = response_body["data"][0]
        self.assertIn("id", model)
        self.assertEqual(model["object"], "model")
        self.assertIn("created", model)

    def test_health_endpoint(self):
        url = f"http://localhost:{self.port}/health"

        response = requests.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

        self.response_generator.stop_and_join()
        response = requests.get(url)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"status": "unavailable"})


class TestServerWithDraftModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(
            DummyModelProvider(with_draft=True), LRUPromptCache()
        )
        cls.server_address = ("localhost", 0)
        cls.httpd = http.server.HTTPServer(
            cls.server_address,
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()

    def test_handle_completions_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/completions"

        post_data = {
            "model": "default_model",
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 0.0,
            "top_p": 1.0,
        }

        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)

        response_body = json.loads(response.text)
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        self.assertIn("usage", response_body)

        # Check that tokens were generated
        self.assertTrue(response_body["usage"]["completion_tokens"] > 0)

    def test_handle_chat_completions_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        response = requests.post(url, json=chat_post_data)
        self.assertEqual(response.status_code, 200)

        response_body = json.loads(response.text)
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        self.assertIn("usage", response_body)

        # Check that tokens were generated
        self.assertTrue(response_body["usage"]["completion_tokens"] > 0)

    def test_streaming_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.0,
            "stream": True,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        response = requests.post(url, json=chat_post_data, stream=True)
        self.assertEqual(response.status_code, 200)

        chunk_count = 0
        for chunk in response.iter_lines():
            if chunk:
                data = chunk.decode("utf-8")
                if data.startswith("data: ") and data != "data: [DONE]":
                    chunk_data = json.loads(data[6:])  # Skip the "data: " prefix
                    self.assertIn("choices", chunk_data)
                    self.assertEqual(len(chunk_data["choices"]), 1)
                    self.assertIn("delta", chunk_data["choices"][0])
                    chunk_count += 1

        # Make sure we got some streaming chunks
        self.assertGreater(chunk_count, 0)

    def test_prompt_cache_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        # First request to initialize cache
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 5,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a story about"},
            ],
        }

        first_response = requests.post(url, json=chat_post_data)
        self.assertEqual(first_response.status_code, 200)

        # Second request with same prefix should use cache
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 5,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a story about dragons."},
            ],
        }

        second_response = requests.post(url, json=chat_post_data)
        self.assertEqual(second_response.status_code, 200)

        # Both responses should have content
        first_response_body = json.loads(first_response.text)
        second_response_body = json.loads(second_response.text)

        self.assertIn("choices", first_response_body)
        self.assertIn("choices", second_response_body)
        self.assertIn("message", first_response_body["choices"][0])
        self.assertIn("message", second_response_body["choices"][0])
        self.assertIn("content", first_response_body["choices"][0]["message"])
        self.assertIn("content", second_response_body["choices"][0]["message"])

        # Ensure both generated content
        self.assertIsNotNone(first_response_body["choices"][0]["message"]["content"])
        self.assertIsNotNone(second_response_body["choices"][0]["message"]["content"])


class TestServerKVCacheQuantization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_provider = DummyModelProvider(kv_bits=4, quantized_kv_start=0)
        cls.prompt_cache = LRUPromptCache()
        cls.response_generator = ResponseGenerator(cls.model_provider, cls.prompt_cache)
        cls.httpd = http.server.HTTPServer(
            ("localhost", 0),
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()

    def test_quantized_kv_disables_batching(self):
        args = type("args", (object,), {"seed": None})
        self.assertFalse(self.response_generator._is_batchable(args))

    def test_completion_quantizes_the_cache(self):
        url = f"http://localhost:{self.port}/v1/completions"
        prompt = "Once upon a time"
        response = requests.post(
            url,
            json={"model": "default_model", "prompt": prompt, "max_tokens": 8},
        )
        self.assertIn("choices", json.loads(response.text))

        tokens = self.model_provider.tokenizer.encode(prompt)
        cache, _ = self.prompt_cache.fetch_nearest_cache(
            self.model_provider.model_key, tokens
        )
        self.assertIsNotNone(cache)
        for c in cache:
            self.assertIsInstance(c, QuantizedKVCache)
            self.assertEqual(c.bits, 4)
            self.assertEqual(c.group_size, 64)


class TestServerWithoutKVCacheQuantization(unittest.TestCase):
    def test_batching_stays_enabled(self):
        prompt_cache = LRUPromptCache()
        response_generator = ResponseGenerator(DummyModelProvider(), prompt_cache)
        try:
            args = type("args", (object,), {"seed": None})
            self.assertTrue(response_generator._is_batchable(args))
        finally:
            response_generator.stop_and_join()


class TestKeepalive(unittest.TestCase):
    def test_keepalive_callback(self):
        """Test keepalive callback sends SSE comments and handles errors"""
        from unittest.mock import Mock

        # Mock handler
        mock_wfile = io.BytesIO()
        handler = Mock()
        handler.wfile = mock_wfile

        # Test callback logic (same as in server.py)
        def keepalive_callback(processed_tokens, total_tokens):
            if handler.stream:
                try:
                    handler.wfile.write(
                        f": keepalive {processed_tokens}/{total_tokens}\n\n".encode()
                    )
                    handler.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        # Test streaming enabled
        handler.stream = True
        keepalive_callback(1024, 4096)

        output = mock_wfile.getvalue().decode("utf-8")
        self.assertEqual(output, ": keepalive 1024/4096\n\n")

        # Test streaming disabled
        handler.stream = False
        mock_wfile.seek(0)
        mock_wfile.truncate(0)
        keepalive_callback(2048, 4096)

        output = mock_wfile.getvalue().decode("utf-8")
        self.assertEqual(output, "")

        # Test error handling
        handler.stream = True
        handler.wfile = Mock()
        handler.wfile.write.side_effect = BrokenPipeError("Connection broken")

        # Should not raise exception
        try:
            keepalive_callback(3072, 4096)
        except Exception as e:
            self.fail(f"Callback should handle BrokenPipeError: {e}")


class TestLRUPromptCache(unittest.TestCase):
    def test_caching(self):
        cache = LRUPromptCache(max_size=10)

        def get_kv(n):
            keys = mx.arange(n).reshape(1, 1, n, 1)
            return keys, keys

        model = ("test", None, None)
        tokens = [10] * 24

        c, t = cache.fetch_nearest_cache(model, tokens)
        self.assertTrue(c is None)
        self.assertEqual(t, tokens)

        c = [KVCache()]
        c[0].update_and_fetch(*get_kv(24))
        cache.insert_cache(model, t, c)

        # Fetching a cache that is strictly a prefix doesn't remove it from the
        # lru cache
        tokens = tokens + [20] * 5
        c, t = cache.fetch_nearest_cache(model, tokens)
        k, v = c[0].keys_and_values()
        self.assertTrue((k == v).all().item())
        self.assertTrue((k.flatten() == mx.arange(24)).all().item())
        self.assertEqual(t, [20] * 5)
        self.assertEqual(len(cache), 1)

        # Inserting a trimmable cache with shared prefix removes the prefixes
        tokens = tokens + [30] * 3
        c[0].update_and_fetch(*get_kv(8))
        cache.insert_cache(model, tokens, c)
        self.assertEqual(len(cache), 1)

        # Fetching a cache with a shared prefix doesn't remove it either
        tokens = tokens[:26] + [40] * 8
        c, t = cache.fetch_nearest_cache(model, tokens)
        k, v = c[0].keys_and_values()
        self.assertTrue((k == v).all().item())
        self.assertTrue(
            (k.flatten() == mx.concatenate([mx.arange(24), mx.arange(2)])).all().item()
        )
        self.assertEqual(t, [40] * 8)
        self.assertEqual(len(cache), 1)

        # Inserting a diverged cache actually creates another entry
        c[0].update_and_fetch(*get_kv(8))
        cache.insert_cache(model, tokens, c)
        self.assertEqual(len(cache), 2)

    def test_lru(self):
        cache = LRUPromptCache(max_size=2)
        model = ("test", None, None)
        cache.insert_cache(model, [1, 2], [MockCache("test1")])
        cache.insert_cache(model, [2, 3], [MockCache("test2")])

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [1])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [1])
        c, t = cache.fetch_nearest_cache(model, [1, 3, 4])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [3, 4])
        c, t = cache.fetch_nearest_cache(model, [2, 3, 4])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [4])
        c, t = cache.fetch_nearest_cache(model, [2, 4, 5])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [4, 5])

        cache.insert_cache(model, [1, 2], [MockCache("test1")])
        cache.insert_cache(model, [2, 3], [MockCache("test2")])
        cache.insert_cache(model, [3, 4], [MockCache("test3")])

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, None)
        self.assertEqual(t, [1, 2])
        c, t = cache.fetch_nearest_cache(model, [2, 3])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, [MockCache("test3")])
        self.assertEqual(t, [])

        cache.insert_cache(model, [4, 5], [MockCache("test4")], cache_type="user")
        c, t = cache.fetch_nearest_cache(model, [2, 3])
        self.assertEqual(c, None)
        self.assertEqual(t, [2, 3])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, [MockCache("test3")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [4, 5])
        self.assertEqual(c, [MockCache("test4")])
        self.assertEqual(t, [])

        cache.insert_cache(model, [5, 6], [MockCache("test5")])
        cache.insert_cache(model, [6, 7], [MockCache("test6")])
        c, t = cache.fetch_nearest_cache(model, [5, 6])
        self.assertEqual(c, None)
        self.assertEqual(t, [5, 6])
        c, t = cache.fetch_nearest_cache(model, [6, 7])
        self.assertEqual(c, [MockCache("test6")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [4, 5])
        self.assertEqual(c, [MockCache("test4")])
        self.assertEqual(t, [])

    def test_insert_trimmable_cache_removes_immediate_prefix(self):
        cache = LRUPromptCache(max_size=10)
        model = ("test", None, None)

        cache.insert_cache(model, [1, 2], [MockCache("ab")])
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 2)

        cache.insert_cache(model, [1, 2, 3], [MockCache("abc")])
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 3)

    def test_insert_empty_tokens_does_not_self_destruct(self):
        cache = LRUPromptCache(max_size=10)
        model = ("test", None, None)

        cache.insert_cache(model, [], [MockCache("root")])
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 4)

        c, t = cache.fetch_nearest_cache(model, [])
        self.assertIsNotNone(c)
        self.assertEqual(t, [])

    def test_fetch_empty_tokens_after_root_eviction(self):
        cache = LRUPromptCache(max_size=10)
        model = ("test", None, None)

        cache.insert_cache(model, [], [MockCache("root")])
        cache.insert_cache(model, [1], [MockCache("a")])

        c, t = cache.fetch_nearest_cache(model, [])
        self.assertIsNone(c)
        self.assertEqual(t, [])

    def test_lru_bytes(self):
        cache = LRUPromptCache(max_size=100, max_bytes=10)
        model = ("test", None, None)

        cache.insert_cache(model, [1, 2], [MockCache("aaa")])
        cache.insert_cache(model, [3, 4], [MockCache("bbb")])
        cache.insert_cache(model, [4, 5], [MockCache("ccc")])
        cache.insert_cache(model, [6, 7], [MockCache("ddd")])

        self.assertEqual(len(cache), 3)
        self.assertEqual(cache.nbytes, 9)

        cache.trim_to(n_bytes=7)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.nbytes, 6)

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, None)
        self.assertEqual(t, [1, 2])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, None)
        self.assertEqual(t, [3, 4])


class TestMakeSampler(unittest.TestCase):
    def test_xtc_special_tokens(self):
        class FakeTokenizer:
            eos_token_ids = [0, 1, 9]

            def encode(self, text, add_special_tokens=False):
                return [3]

        sampling = SamplingArguments(
            temperature=0.6,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            xtc_probability=1.0,
            xtc_threshold=0.1,
        )
        args = type("obj", (object,), {"sampling": sampling})
        sampler = _make_sampler(args, FakeTokenizer())
        logits = mx.log(
            mx.array([[0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02]])
        )
        token = sampler(logits)
        mx.eval(token)
        self.assertEqual(token.shape, (1,))


class TestModelSwapClearsCache(unittest.TestCase):
    @mock.patch("mlx_lm.server.mx.clear_cache")
    @mock.patch("mlx_lm.server.make_prompt_cache", return_value=[])
    @mock.patch("mlx_lm.server.load")
    @mock.patch("mlx_lm.server.mx.distributed.init")
    def test_load_clears_mlx_buffer_pool(self, init, load, make_cache, clear_cache):
        import argparse

        from mlx_lm.server import ModelProvider

        args = argparse.Namespace(
            adapter_path=None,
            chat_template=None,
            draft_model=None,
            model="model-a",
            pipeline=False,
            trust_remote_code=False,
            use_default_chat_template=False,
        )
        tokenizer = mock.Mock()
        tokenizer.chat_template = None
        tokenizer.default_chat_template = None
        load.return_value = (mock.Mock(), tokenizer, {})
        init.return_value.size.return_value = 1

        provider = ModelProvider(args)
        provider._load("model-a")
        provider._load("model-b")
        self.assertGreaterEqual(clear_cache.call_count, 2)


class FailingModelProvider:
    def load_default(self):
        raise RuntimeError("simulated generate crash")


class TestGenerationThreadDeath(unittest.TestCase):
    def _crashed_generator(self):
        rg = ResponseGenerator(FailingModelProvider(), LRUPromptCache())
        rg.join()
        time.sleep(0.05)
        return rg

    def test_generation_unavailable_after_thread_crash(self):
        rg = self._crashed_generator()
        self.assertTrue(rg._generation_failed)
        self.assertFalse(rg.generation_available())
        with self.assertRaisesRegex(RuntimeError, "generation thread died"):
            rg.generate(None, None)

    def test_inflight_request_does_not_hang_after_thread_crash(self):
        rg = self._crashed_generator()
        # A request dequeued before the crash never gets a response queued, so
        # waiting on it must give up instead of blocking forever.
        with self.assertRaisesRegex(RuntimeError, "generation thread died"):
            rg._await_response(Queue())


class TestIgnoreEos(unittest.TestCase):
    """`ignore_eos` through the HTTP handler and the real generation thread.

    The model is real; its choices are scripted with a logits processor, so a request
    emits a known token sequence (EOS tokens included) on both the batched path and
    the serial path (a request with a seed).
    """

    PROVIDER_KWARGS = {}

    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(
            DummyModelProvider(**cls.PROVIDER_KWARGS), LRUPromptCache()
        )
        cls.httpd = http.server.HTTPServer(
            ("localhost", 0),
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()
        tokenizer = cls.response_generator.model_provider.tokenizer
        cls.eos = sorted(tokenizer.eos_token_ids)[0]
        cls.hello = tokenizer.encode("hello", add_special_tokens=False)
        cls.world = tokenizer.encode(" world", add_special_tokens=False)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()

    @staticmethod
    def forced(sequence):
        """Logits processors that pick sequence[i] at step i, then the last."""

        def make(args):
            step = 0

            def processor(tokens, logits):
                nonlocal step
                token = sequence[min(step, len(sequence) - 1)]
                step += 1
                return logits + mx.zeros(logits.shape[-1]).at[token].add(2000.0)

            return [processor]

        return make

    def send(
        self,
        endpoint="chat",
        stream=False,
        seed=None,
        max_tokens=8,
        limit_key="max_tokens",
        stream_options=None,
        **extra,
    ):
        """Returns (text, finish_reason, completion_tokens) for one request."""
        body = {"model": "chat_model", limit_key: max_tokens, **extra}
        if endpoint == "chat":
            path = "/v1/chat/completions"
            body["messages"] = [{"role": "user", "content": "Hi"}]
        else:
            path = "/v1/completions"
            body["prompt"] = "Hi"
        if seed is not None:
            body["seed"] = seed
        if stream:
            body["stream"] = True
            body["stream_options"] = stream_options or {"include_usage": True}
        r = requests.post(
            f"http://localhost:{self.port}{path}", json=body, stream=stream
        )
        self.assertEqual(r.status_code, 200, r.text)
        if not stream:
            choice = r.json()["choices"][0]
            message = choice.get("message")
            text = choice["text"] if endpoint == "text" else message["content"]
            usage = r.json()["usage"]
            return text or "", choice["finish_reason"], usage["completion_tokens"]
        text, finish, usage = "", None, None
        for line in r.iter_lines():
            if not line.startswith(b"data: ") or line.endswith(b"[DONE]"):
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            for choice in chunk["choices"]:
                if endpoint == "text":
                    text += choice["text"]
                else:
                    text += choice["delta"].get("content") or ""
                finish = choice.get("finish_reason") or finish
        return text, finish, usage["completion_tokens"]

    def request(self, sequence, **kwargs):
        # The generation thread reads the patched name when it takes the request,
        # so the patch stays until the whole response is read.
        with mock.patch("mlx_lm.server._make_logits_processors", self.forced(sequence)):
            return self.send(**kwargs)

    def post_json(self, sequence, path, body):
        """The JSON of one non-streaming request with the scripted token choices."""
        with mock.patch("mlx_lm.server._make_logits_processors", self.forced(sequence)):
            r = requests.post(f"http://localhost:{self.port}{path}", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def every_path(self):
        for endpoint in ("chat", "text"):
            for stream in (False, True):
                for seed in (None, 7):
                    yield dict(endpoint=endpoint, stream=stream, seed=seed)

    def test_default_stops_at_the_first_eos(self):
        for path in self.every_path():
            with self.subTest(**path):
                self.assertEqual(self.request([self.eos], **path), ("", "stop", 1))

    def test_false_and_null_are_the_default(self):
        for value in (False, None):
            for path in self.every_path():
                with self.subTest(ignore_eos=value, **path):
                    got = self.request([self.eos], ignore_eos=value, **path)
                    self.assertEqual(got, ("", "stop", 1))

    def test_ignore_eos_runs_to_max_tokens(self):
        for max_tokens in (1, 2, 8):
            for path in self.every_path():
                with self.subTest(max_tokens=max_tokens, **path):
                    got = self.request(
                        [self.eos], ignore_eos=True, max_tokens=max_tokens, **path
                    )
                    # EOS tokens count and add no text; the limit ends the request
                    self.assertEqual(got, ("", "length", max_tokens))

    def test_ignore_eos_keeps_the_text_around_eos(self):
        sequence = [*self.hello, self.eos, *self.world]
        for path in self.every_path():
            with self.subTest(**path):
                got = self.request(
                    sequence, ignore_eos=True, max_tokens=len(sequence), **path
                )
                self.assertEqual(got, ("hello world", "length", len(sequence)))
                # without it the request ends at the EOS
                got = self.request(sequence, **path)
                self.assertEqual(got, ("hello", "stop", len(self.hello) + 1))

    def test_stop_strings_still_stop_when_eos_is_ignored(self):
        sequence = [*self.hello, self.eos, *self.world, *self.hello]
        n = len(self.hello) + 1 + len(self.world)
        for path in self.every_path():
            with self.subTest(**path):
                got = self.request(
                    sequence, ignore_eos=True, stop=[" world"], max_tokens=32, **path
                )
                self.assertEqual(got, ("hello", "stop", n))

    def test_streaming_matches_non_streaming(self):
        sequence = [*self.hello, self.eos, *self.world, self.eos]
        for endpoint in ("chat", "text"):
            for seed in (None, 7):
                for ignore_eos in (False, True):
                    with self.subTest(endpoint=endpoint, seed=seed, eos=ignore_eos):
                        got = [
                            self.request(
                                sequence,
                                endpoint=endpoint,
                                stream=stream,
                                seed=seed,
                                ignore_eos=ignore_eos,
                                max_tokens=len(sequence),
                            )
                            for stream in (False, True)
                        ]
                        self.assertEqual(got[0], got[1])

    def test_requests_with_and_without_it_do_not_share_state(self):
        # the stop state machine is cached per model and stop words
        for _ in range(2):
            self.assertEqual(
                self.request([self.eos], ignore_eos=True), ("", "length", 8)
            )
            self.assertEqual(self.request([self.eos]), ("", "stop", 1))

    def test_concurrent_requests_each_get_their_own_setting(self):
        results = {}

        def run(i):
            results[i] = self.send(ignore_eos=i % 2 == 0, stream=True)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(6)]
        forced = self.forced([self.eos])
        with mock.patch("mlx_lm.server._make_logits_processors", forced):
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        for i in range(6):
            want = ("", "length", 8) if i % 2 == 0 else ("", "stop", 1)
            self.assertEqual(results[i], want)

    def test_non_boolean_ignore_eos_is_rejected(self):
        for value in ("yes", "false", "true", 0, 1, [True]):
            r = requests.post(
                f"http://localhost:{self.port}/v1/completions",
                json={"prompt": "hi", "max_tokens": 2, "ignore_eos": value},
            )
            self.assertEqual(r.status_code, 400, value)
            self.assertIn("ignore_eos", r.json()["error"])

    def test_the_body_guidellm_sends(self):
        # guidellm 0.8.1: top-level ignore_eos with stop null; the limit is
        # max_completion_tokens on chat and max_tokens on text completions.
        options = {"include_usage": True, "continuous_usage_stats": True}
        limits = (("chat", "max_completion_tokens"), ("text", "max_tokens"))
        for endpoint, limit_key in limits:
            for stream in (False, True):
                for seed in (None, 7):
                    with self.subTest(endpoint=endpoint, stream=stream, seed=seed):
                        got = self.request(
                            [self.eos],
                            endpoint=endpoint,
                            stream=stream,
                            seed=seed,
                            limit_key=limit_key,
                            max_tokens=5,
                            stream_options=options,
                            stop=None,
                            ignore_eos=True,
                        )
                        self.assertEqual(got, ("", "length", 5))

    def test_logprobs_cover_the_ignored_eos_tokens(self):
        sequence = [*self.hello, self.eos, *self.world]
        for endpoint, path in (
            ("chat", "/v1/chat/completions"),
            ("text", "/v1/completions"),
        ):
            for seed in (None, 7):
                with self.subTest(endpoint=endpoint, seed=seed):
                    body = {
                        "model": "chat_model",
                        "max_tokens": len(sequence),
                        "ignore_eos": True,
                        "logprobs": True,
                    }
                    if endpoint == "chat":
                        body["messages"] = [{"role": "user", "content": "Hi"}]
                    else:
                        body["prompt"] = "Hi"
                    if seed is not None:
                        body["seed"] = seed
                    choice = self.post_json(sequence, path, body)["choices"][0]
                    self.assertEqual(choice["finish_reason"], "length")
                    self.assertEqual(
                        [e["id"] for e in choice["logprobs"]["content"]], sequence
                    )

    def test_cache_holds_the_continuation_past_an_ignored_eos(self):
        tokenizer = self.response_generator.model_provider.tokenizer
        sequence = [self.eos, *self.hello]
        for seed in (None, 7):
            with self.subTest(seed=seed):
                # Unrelated prompts, so the first request starts with an empty cache.
                prompt = "Alpha probe" if seed is None else "Zulu probe"
                body = {
                    "model": "chat_model",
                    "prompt": prompt,
                    "max_tokens": len(sequence),
                    "ignore_eos": True,
                }
                if seed is not None:
                    body["seed"] = seed
                first = self.post_json(sequence, "/v1/completions", body)
                details = first["usage"]["prompt_tokens_details"]
                self.assertEqual(details["cached_tokens"], 0)
                n = first["usage"]["prompt_tokens"] + len(sequence)
                # A prompt that continues past the EOS reuses the whole continuation.
                follow_up = prompt + tokenizer.decode(sequence) + " world"
                tokens = tokenizer.encode(follow_up, add_special_tokens=False)
                self.assertEqual(tokens[:n], tokenizer.encode(prompt) + sequence)
                body = dict(body, prompt=follow_up, max_tokens=1)
                second = self.post_json(self.world, "/v1/completions", body)
                self.assertEqual(second["usage"]["prompt_tokens"], len(tokens))
                details = second["usage"]["prompt_tokens_details"]
                self.assertEqual(details["cached_tokens"], n)

    @contextlib.contextmanager
    def tool_parser(self, module):
        """The server's tokenizer with the tool parser of `module`."""
        tokenizer = self.response_generator.model_provider.tokenizer
        cache = self.response_generator._state_machine_cache
        with (
            mock.patch.object(tokenizer, "_tool_parser", module.parse_tool_call),
            mock.patch.object(tokenizer, "_tool_call_start", module.tool_call_start),
            mock.patch.object(tokenizer, "_tool_call_end", module.tool_call_end),
        ):
            # The cached state machines belong to the tokenizer without it.
            cache.clear()
            try:
                yield tokenizer
            finally:
                cache.clear()

    def tool_request(self, module, sequence, **extra):
        """(finish_reason, content, tool call names) of a chat request with tools."""
        body = {
            "model": "chat_model",
            "max_tokens": len(sequence),
            "messages": [{"role": "user", "content": "Weather in Tokyo?"}],
            "tools": ScriptedHandler.TOOLS,
            **extra,
        }
        with self.tool_parser(module):
            response = self.post_json(sequence, "/v1/chat/completions", body)
        choice = response["choices"][0]
        message = choice["message"]
        calls = [c["function"]["name"] for c in message.get("tool_calls") or []]
        return choice["finish_reason"], message["content"], calls

    def test_a_call_that_ends_at_eos_is_kept_when_eos_is_the_last_token(self):
        # llama3 has no end marker: the call text ends where the EOS would be.
        encode = self.response_generator.model_provider.tokenizer.encode
        call = encode(
            '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Tokyo"}}',
            add_special_tokens=False,
        )
        self.assertEqual(
            self.tool_request(llama3, [*call, self.eos]),
            ("tool_calls", None, ["get_weather"]),
        )
        self.assertEqual(
            self.tool_request(llama3, [*call, self.eos], ignore_eos=True),
            ("length", None, ["get_weather"]),
        )

    def test_text_after_an_ignored_eos_joins_a_call_without_an_end_marker(self):
        encode = self.response_generator.model_provider.tokenizer.encode
        call = encode(
            '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Tokyo"}}',
            add_special_tokens=False,
        )
        more = encode("\n\nHello there", add_special_tokens=False)
        finish, content, calls = self.tool_request(
            llama3, [*call, self.eos, *more], ignore_eos=True
        )
        # The call text is no longer only a call, so it is returned as content.
        self.assertEqual((finish, calls), ("length", []))
        self.assertEqual(
            content,
            '{"name": "get_weather", "parameters": {"city": "Tokyo"}}\n\nHello there',
        )

    def test_a_call_with_an_end_marker_is_kept_when_text_follows_an_ignored_eos(self):
        encode = self.response_generator.model_provider.tokenizer.encode
        call = encode(
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n"
            "</parameter>\n</function>\n</tool_call>",
            add_special_tokens=False,
        )
        more = encode(" Done", add_special_tokens=False)
        self.assertEqual(
            self.tool_request(qwen3_coder, [*call, self.eos, *more], ignore_eos=True),
            ("length", " Done", ["get_weather"]),
        )


class TestIgnoreEosWithDraftModel(TestIgnoreEos):
    """The same requests with a draft model, so every request is speculative.

    Only the serial path takes a draft model. A seed makes a request serial, so every
    request carries one: a serial request that reuses a cache made by a batched
    request has no draft cache part and fails, with or without ignore_eos.
    """

    PROVIDER_KWARGS = {"with_draft": True}

    # A scripted token sequence counts the calls of the logits processor, and a
    # speculative step calls it for draft tokens too. Only the tests whose model
    # keeps choosing the EOS token (a constant sequence) run here.
    test_ignore_eos_keeps_the_text_around_eos = None
    test_stop_strings_still_stop_when_eos_is_ignored = None
    test_streaming_matches_non_streaming = None
    test_logprobs_cover_the_ignored_eos_tokens = None
    test_cache_holds_the_continuation_past_an_ignored_eos = None
    test_a_call_that_ends_at_eos_is_kept_when_eos_is_the_last_token = None
    test_text_after_an_ignored_eos_joins_a_call_without_an_end_marker = None
    test_a_call_with_an_end_marker_is_kept_when_text_follows_an_ignored_eos = None

    def send(self, *args, seed=None, **kwargs):
        return super().send(*args, seed=7 if seed is None else seed, **kwargs)

    def post_json(self, sequence, path, body):
        return super().post_json(sequence, path, {"seed": 7, **body})


if __name__ == "__main__":
    unittest.main()
