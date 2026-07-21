"""Unit tests for the Anthropic <-> OpenAI conversion layer.

Covers request conversion (system/tools/tool_choice/images/tool_result
ordering, stream_options), response conversion (stop reasons, usage,
reasoning), and the streaming state machine driven by scripted chunk
sequences (text -> tool_calls, split tool-call deltas, usage-only final
chunk, missing finish_reason). Pure/synchronous — no network or provider.
"""

import json
import unittest

from app.anthropic_models import AnthropicMessagesRequest
from app.conversion.anthropic_openai import (
    AnthropicToOpenAIConverter,
    AnthropicToResponsesConverter,
    OpenAIToAnthropicConverter,
    StreamConversionState,
)


def _parse_sse(sse_output):
    """Flatten SSE string(s) into a list of (event_type, data_dict)."""
    if isinstance(sse_output, str):
        sse_output = [sse_output]
    events = []
    for chunk in sse_output:
        for block in chunk.strip().split("\n\n"):
            block = block.strip()
            if not block:
                continue
            etype = None
            data = None
            for line in block.split("\n"):
                if line.startswith("event: "):
                    etype = line[len("event: "):]
                elif line.startswith("data: "):
                    data = json.loads(line[len("data: "):])
            events.append((etype, data))
    return events


def _req(**extra):
    base = {
        "model": "claude-x",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    }
    base.update(extra)
    return AnthropicMessagesRequest(**base)


# ---------------------------------------------------------------------------
# Anthropic -> OpenAI request conversion
# ---------------------------------------------------------------------------

class ConvertRequestTests(unittest.TestCase):
    def setUp(self):
        self.c = AnthropicToOpenAIConverter()

    def test_system_string_becomes_system_message(self):
        out = self.c.convert_request(_req(system="be nice"), "m")
        self.assertEqual(out.messages[0].role, "system")
        self.assertEqual(out.messages[0].content, "be nice")

    def test_system_list_is_concatenated(self):
        req = _req(system=[
            {"type": "text", "text": "line one"},
            {"type": "text", "text": "line two"},
        ])
        out = self.c.convert_request(req, "m")
        self.assertEqual(out.messages[0].content, "line one\nline two")

    def test_tools_converted_to_openai_function_format(self):
        req = _req(tools=[{
            "name": "get_weather",
            "description": "gets weather",
            "input_schema": {"type": "object", "properties": {"loc": {"type": "string"}}},
        }])
        out = self.c.convert_request(req, "m")
        self.assertEqual(out.tools[0].type, "function")
        self.assertEqual(out.tools[0].function.name, "get_weather")
        self.assertEqual(
            out.tools[0].function.parameters,
            {"type": "object", "properties": {"loc": {"type": "string"}}},
        )

    def test_server_tool_filtered_out(self):
        req = _req(tools=[
            {"type": "web_search_20250305", "name": "web_search_x"},
            {"name": "real", "input_schema": {"type": "object"}},
        ])
        out = self.c.convert_request(req, "m")
        names = [t.function.name for t in out.tools]
        self.assertEqual(names, ["real"])

    def test_tool_choice_any_maps_to_required(self):
        out = self.c.convert_request(_req(tool_choice={"type": "any"}), "m")
        self.assertEqual(out.tool_choice, "required")

    def test_tool_choice_specific_tool(self):
        out = self.c.convert_request(_req(tool_choice={"type": "tool", "name": "f"}), "m")
        self.assertEqual(out.tool_choice, {"type": "function", "function": {"name": "f"}})

    def test_image_block_base64_to_data_url(self):
        req = _req(messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
            ],
        }])
        out = self.c.convert_request(req, "m")
        user_msg = out.messages[-1]
        part = user_msg.content[0]
        self.assertEqual(part.type, "image_url")
        self.assertEqual(part.image_url.url, "data:image/png;base64,AAAA")

    def test_tool_result_emitted_before_user_text(self):
        # assistant(tool_calls) -> user turn carrying [tool_result, text].
        # The converted sequence must place the tool message before the user text.
        req = _req(messages=[
            {"role": "user", "content": "run it"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "42"},
                {"type": "text", "text": "thanks"},
            ]},
        ])
        out = self.c.convert_request(req, "m")
        roles = [m.role for m in out.messages]
        # ... user, assistant(tool_calls), tool, user(text)
        self.assertEqual(roles[-3:], ["assistant", "tool", "user"])
        tool_msg = out.messages[-2]
        self.assertEqual(tool_msg.tool_call_id, "toolu_1")
        self.assertEqual(tool_msg.content, "42")

    def test_stream_options_set_when_streaming(self):
        out = self.c.convert_request(_req(stream=True), "m")
        self.assertIsNotNone(out.stream_options)
        self.assertTrue(out.stream_options.include_usage)

    def test_stream_options_absent_when_not_streaming(self):
        out = self.c.convert_request(_req(stream=False), "m")
        self.assertIsNone(out.stream_options)

    def test_sampling_params_only_forwarded_when_set(self):
        out = self.c.convert_request(_req(), "m")
        # Anthropic client sent nothing -> OpenAI request must not carry values.
        self.assertIsNone(out.temperature)
        self.assertIsNone(out.top_p)
        out2 = self.c.convert_request(_req(temperature=0.5), "m")
        self.assertEqual(out2.temperature, 0.5)


class ConvertResponsesRequestOrderingTests(unittest.TestCase):
    def test_function_call_output_before_user_text_responses_path(self):
        c = AnthropicToResponsesConverter()
        req = _req(messages=[
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
                {"type": "text", "text": "thanks"},
            ]},
        ])
        out = c.convert_request(req, "codex")
        types = [item.get("type") for item in out.input]
        # function_call, function_call_output, message(user text)
        self.assertEqual(types, ["function_call", "function_call_output", "message"])


# ---------------------------------------------------------------------------
# OpenAI -> Anthropic response conversion
# ---------------------------------------------------------------------------

class ConvertResponseTests(unittest.TestCase):
    def setUp(self):
        self.c = OpenAIToAnthropicConverter()

    def _resp(self, message, finish_reason="stop", usage=None):
        return {
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": usage or {"prompt_tokens": 7, "completion_tokens": 3},
        }

    def test_text_and_usage(self):
        out = self.c.convert_response(self._resp({"content": "hello"}), "claude-x")
        self.assertEqual(out["content"], [{"type": "text", "text": "hello"}])
        self.assertEqual(out["usage"], {"input_tokens": 7, "output_tokens": 3})
        self.assertEqual(out["stop_reason"], "end_turn")
        self.assertEqual(out["model"], "claude-x")

    def test_stop_reason_mapping(self):
        cases = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "refusal",
            "weird_new_reason": "end_turn",
        }
        for finish, expected in cases.items():
            out = self.c.convert_response(self._resp({"content": "x"}, finish_reason=finish), "m")
            self.assertEqual(out["stop_reason"], expected, finish)

    def test_tool_calls_converted(self):
        msg = {
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "function": {"name": "f", "arguments": '{"a": 1}'},
            }],
        }
        out = self.c.convert_response(self._resp(msg, finish_reason="tool_calls"), "m")
        tu = [b for b in out["content"] if b["type"] == "tool_use"][0]
        self.assertEqual(tu["id"], "call_1")
        self.assertEqual(tu["name"], "f")
        self.assertEqual(tu["input"], {"a": 1})

    def test_reasoning_becomes_thinking_block(self):
        msg = {"reasoning_content": "let me think", "content": "answer"}
        out = self.c.convert_response(self._resp(msg), "m")
        self.assertEqual(out["content"][0], {"type": "thinking", "thinking": "let me think", "signature": ""})
        self.assertEqual(out["content"][1], {"type": "text", "text": "answer"})

    def test_empty_content_yields_empty_text_block(self):
        out = self.c.convert_response(self._resp({"content": None}), "m")
        self.assertEqual(out["content"], [{"type": "text", "text": ""}])


# ---------------------------------------------------------------------------
# Streaming converter (OpenAI chunks -> Anthropic SSE)
# ---------------------------------------------------------------------------

class StreamConverterTests(unittest.TestCase):
    def setUp(self):
        self.c = OpenAIToAnthropicConverter()
        self.state = StreamConversionState(message_id="msg_1", model="m")

    def _feed(self, chunk):
        return _parse_sse(self.c.convert_stream_chunk(chunk, self.state))

    def _text_chunk(self, text, finish=None):
        return {"choices": [{"delta": {"content": text}, "finish_reason": finish}]}

    def test_text_then_tool_calls_sequence(self):
        # Text delta opens a text block.
        events = self._feed(self._text_chunk("Hi"))
        types = [e[0] for e in events]
        self.assertEqual(types, ["content_block_start", "content_block_delta"])
        self.assertEqual(events[0][1]["content_block"]["type"], "text")

        # A tool call (id + name in one delta) closes text, opens tool_use.
        tc_chunk = {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "call_1",
            "function": {"name": "f", "arguments": '{"a"'},
        }]}, "finish_reason": None}]}
        events = self._feed(tc_chunk)
        types = [e[0] for e in events]
        self.assertEqual(types, ["content_block_stop", "content_block_start", "content_block_delta"])
        self.assertEqual(events[1][1]["content_block"], {
            "type": "tool_use", "id": "call_1", "name": "f", "input": {},
        })
        self.assertEqual(events[2][1]["delta"], {"type": "input_json_delta", "partial_json": '{"a"'})
        self.assertTrue(self.state.had_tool_use)

        # Finish + end: stop_reason is tool_use even without tool_calls finish.
        self._feed(self._text_chunk("", finish="stop"))
        end = _parse_sse(self.c.convert_stream_end(self.state))
        delta = [e for e in end if e[0] == "message_delta"][0]
        self.assertEqual(delta[1]["delta"]["stop_reason"], "tool_use")

    def test_split_tool_call_deltas_buffer_arguments(self):
        # id arrives first (no name) — no block should open, args buffered.
        e1 = self._feed({"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "call_9", "function": {"arguments": '{"x'},
        }]}}]})
        self.assertEqual(e1, [])  # nothing emitted; index has no opened block
        self.assertNotIn(0, self.state.tool_call_indices)

        # name arrives — block opens and buffered args flush.
        e2 = self._feed({"choices": [{"delta": {"tool_calls": [{
            "index": 0, "function": {"name": "f", "arguments": ':1}'},
        }]}}]})
        types = [e[0] for e in e2]
        self.assertEqual(types, ["content_block_start", "content_block_delta", "content_block_delta"])
        self.assertEqual(e2[0][1]["content_block"]["id"], "call_9")
        self.assertEqual(e2[0][1]["content_block"]["name"], "f")
        # buffered fragment first, then the new fragment
        self.assertEqual(e2[1][1]["delta"]["partial_json"], '{"x')
        self.assertEqual(e2[2][1]["delta"]["partial_json"], ':1}')

    def test_usage_only_final_chunk_populates_tokens(self):
        self._feed(self._text_chunk("hi"))
        # A trailing usage-only chunk (no choices) — from stream_options.
        self._feed({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 4}})
        self.assertEqual(self.state.input_tokens, 11)
        self.assertEqual(self.state.output_tokens, 4)
        end = _parse_sse(self.c.convert_stream_end(self.state))
        delta = [e for e in end if e[0] == "message_delta"][0]
        self.assertEqual(delta[1]["usage"], {"input_tokens": 11, "output_tokens": 4})

    def test_missing_finish_reason_defaults_end_turn_for_text(self):
        self._feed(self._text_chunk("hello"))
        end = _parse_sse(self.c.convert_stream_end(self.state))
        delta = [e for e in end if e[0] == "message_delta"][0]
        self.assertEqual(delta[1]["delta"]["stop_reason"], "end_turn")

    def test_no_content_emits_empty_text_block(self):
        end = _parse_sse(self.c.convert_stream_end(self.state))
        types = [e[0] for e in end]
        self.assertEqual(types[:2], ["content_block_start", "content_block_stop"])
        self.assertEqual(types[-2:], ["message_delta", "message_stop"])


if __name__ == "__main__":
    unittest.main()
