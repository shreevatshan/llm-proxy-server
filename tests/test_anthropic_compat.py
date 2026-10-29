"""Anthropic Messages compatibility for OpenAI-only upstreams.

Covers the request/response/stream converters' fidelity features (documents,
tool-result media, thinking, tool-name mapping, stop sequences, citations,
interleaved tool calls), the adapter's error shaping and rejected-parameter
retry, custom-provider routing (adapter vs native), and web search running
through the real adapter path.
"""

import asyncio
import json
import unittest
from unittest.mock import patch

from app.anthropic_models import AnthropicMessagesRequest
from app.conversion.anthropic_helpers import (
    ThinkTagSplitter,
    ToolNameMap,
    parse_tool_arguments,
    reasoning_effort_for,
    restore_tool_id,
    tool_id_signature,
)
from app.conversion.anthropic_openai import (
    AnthropicToOpenAIConverter,
    AnthropicToResponsesConverter,
    OpenAIToAnthropicConverter,
    ResponsesToAnthropicConverter,
    StreamConversionState,
)
from app.conversion.token_estimate import estimate_anthropic_input_tokens
from app.openai_models import ChatCompletionResponse
from app.providers import anthropic_adapter
from app.providers.anthropic_adapter import (
    anthropic_adapter_messages,
    anthropic_adapter_messages_stream,
    prepare_anthropic_adapter_request,
)
from app.providers.base import ProviderHTTPError
from app.providers.custom_providers import create_custom_provider
from app.websearch import WebSearchRequestError, backends, maybe_intercept_anthropic, tools
from app.websearch.results import SearchResult, SearchSucceeded
from app.websearch.settings import WebSearchConfig, websearch_settings_cache


def run(coro):
    return asyncio.run(coro)


def _req(**extra):
    body = {"model": "custom:t/m", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}
    body.update(extra)
    return AnthropicMessagesRequest.model_validate(body)


def _parse_sse(chunks):
    events = []
    for chunk in chunks if isinstance(chunks, list) else [chunks]:
        for raw in chunk.strip().split("\n\n"):
            data = None
            etype = None
            for line in raw.splitlines():
                if line.startswith("event: "):
                    etype = line[7:]
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
            if etype:
                events.append((etype, data))
    return events


def _messages_dump(out):
    return out.model_dump(exclude_none=True)["messages"]


# ---------------------------------------------------------------------------
# Request conversion
# ---------------------------------------------------------------------------

class RequestConversionTests(unittest.TestCase):
    def setUp(self):
        self.c = AnthropicToOpenAIConverter(echo_reasoning=True)

    def test_pdf_document_becomes_file_part(self):
        req = _req(messages=[{"role": "user", "content": [
            {"type": "document", "title": "Report", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="}},
            {"type": "text", "text": "summarize"},
        ]}])
        parts = _messages_dump(self.c.convert_request(req, "m"))[0]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "Document: Report"})
        self.assertEqual(parts[1], {"type": "file", "file": {"filename": "Report.pdf", "file_data": "data:application/pdf;base64,JVBERi0="}})
        self.assertEqual(parts[2], {"type": "text", "text": "summarize"})

    def test_pdf_document_placeholder_without_file_part_support(self):
        req = _req(messages=[{"role": "user", "content": [
            {"type": "document", "title": "Report", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="}},
        ]}])
        out = AnthropicToOpenAIConverter(supports_file_parts=False).convert_request(req, "m")
        parts = _messages_dump(out)[0]["content"]
        self.assertEqual([p["type"] for p in parts], ["text", "text"])
        self.assertEqual(parts[1]["text"], "[document Report (application/pdf) is not available to this model]")

    def test_gemini_thought_signature_tool_id_round_trips(self):
        raw_id = "call_x__thought__Q2lnbmF0dXJl+/A="
        out = OpenAIToAnthropicConverter().convert_response(
            _chat_response({"content": None, "tool_calls": [{"id": raw_id, "type": "function", "function": {"name": "f", "arguments": "{}"}}]}, "tool_calls").model_dump(),
            "m",
        )
        tool_id = out["content"][0]["id"]
        self.assertRegex(tool_id, r"^[a-zA-Z0-9_-]+$")
        req = _req(messages=[
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": tool_id, "name": "f", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}]},
        ])
        sanitized, _ = prepare_anthropic_adapter_request(req, transport="chat")
        msgs = _messages_dump(AnthropicToOpenAIConverter().convert_request(sanitized, "m"))
        self.assertEqual(msgs[1]["tool_calls"][0]["id"], raw_id)
        self.assertEqual(msgs[2]["tool_call_id"], raw_id)

    def _replay(self, tool_uses, converter):
        req = _req(messages=[
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": i, "name": "f", "input": {}} for i in tool_uses]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": i, "content": "ok"} for i in tool_uses]},
        ])
        sanitized, _ = prepare_anthropic_adapter_request(req, transport="chat")
        return _messages_dump(converter.convert_request(sanitized, "m"))

    def test_gemini_extra_content_thought_signature_round_trips(self):
        # Gemini signs only the first of parallel calls, beside the id.
        sig = "CiIBjz1rX+/sig=="
        out = OpenAIToAnthropicConverter().convert_response({"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"},
                 "extra_content": {"google": {"thought_signature": sig}}},
                {"id": "c2", "type": "function", "function": {"name": "f", "arguments": "{}"}},
            ]}}]}, "m")
        ids = [b["id"] for b in out["content"]]
        self.assertRegex(ids[0], r"^[a-zA-Z0-9_-]+$")
        self.assertEqual(ids[1], "c2")

        msgs = self._replay(ids, AnthropicToOpenAIConverter(thought_signatures=True))
        calls = msgs[1]["tool_calls"]
        self.assertEqual([c["id"] for c in calls], ["c1", "c2"])
        self.assertEqual(calls[0]["extra_content"], {"google": {"thought_signature": sig}})
        self.assertNotIn("extra_content", calls[1])
        self.assertEqual([m["tool_call_id"] for m in msgs[2:]], ["c1", "c2"])

        # Other OpenAI-compatible upstreams get the bare id and no extension.
        msgs = self._replay(ids, AnthropicToOpenAIConverter())
        self.assertEqual(msgs[1]["tool_calls"][0]["id"], "c1")
        self.assertNotIn("extra_content", msgs[1]["tool_calls"][0])

    def test_unsigned_tool_calls_get_skip_signature_for_gemini(self):
        msgs = self._replay(["toolu_a", "toolu_b"], AnthropicToOpenAIConverter(thought_signatures=True))
        calls = msgs[1]["tool_calls"]
        self.assertEqual(calls[0]["extra_content"], {"google": {"thought_signature": "skip_thought_signature_validator"}})
        self.assertNotIn("extra_content", calls[1])

    def _tool_search_result(self, content):
        req = _req(messages=[
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "ToolSearch", "input": {"query": "x"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": content}]},
        ])
        return _messages_dump(self.c.convert_request(req, "m"))[2]["content"]

    def test_tool_reference_becomes_text(self):
        with self.assertNoLogs("app.conversion.anthropic_openai", level="WARNING"):
            text = self._tool_search_result([
                {"type": "tool_reference", "tool_name": "a"},
                {"type": "tool_reference", "tool_name": "b"},
            ])
        self.assertEqual(text, "Tool loaded: a\nTool loaded: b")

    def test_tool_reference_uses_the_declared_tool_name(self):
        long_name = "mcp__" + "x" * 80
        text = self._tool_search_result([{"type": "tool_reference", "tool_name": long_name}])
        self.assertEqual(text, f"Tool loaded: {ToolNameMap().to_openai(long_name)}")

    def test_tool_reference_keeps_surrounding_text(self):
        text = self._tool_search_result([
            {"type": "text", "text": "Found 1 tool"},
            {"type": "tool_reference", "tool_name": "a"},
        ])
        self.assertEqual(text, "Found 1 tool\nTool loaded: a")

    def test_responses_keeps_unpaired_server_tool_use(self):
        req = _req(messages=[
            {"role": "user", "content": "run it"},
            {"role": "assistant", "content": [
                {"type": "server_tool_use", "id": "srvtoolu_1", "name": "code_execution", "input": {"code": "print(1)"}},
                {"type": "text", "text": "done"},
            ]},
        ])
        sanitized, _ = prepare_anthropic_adapter_request(req, transport="responses")
        out = AnthropicToResponsesConverter().convert_request(sanitized, "codex").model_dump(exclude_none=True)
        self.assertIn("code_execution", json.dumps(out["input"]))

    def test_text_document_and_search_result(self):
        req = _req(messages=[{"role": "user", "content": [
            {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "doc body"}},
            {"type": "search_result", "source": "https://s.example", "title": "S", "content": [{"type": "text", "text": "found"}]},
        ]}])
        parts = _messages_dump(self.c.convert_request(req, "m"))[0]["content"]
        self.assertEqual(parts[0]["text"], "doc body")
        self.assertIn("https://s.example", parts[1]["text"])
        self.assertIn("found", parts[1]["text"])

    def test_tool_result_media_moves_to_follow_up_user_message(self):
        req = _req(messages=[
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "shot", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "is_error": True, "content": [
                    {"type": "text", "text": "partial"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBO"}},
                ]},
                {"type": "text", "text": "continue"},
            ]},
        ])
        msgs = _messages_dump(self.c.convert_request(req, "m"))
        tool_msg = msgs[2]
        self.assertEqual(tool_msg, {"role": "tool", "tool_call_id": "toolu_1", "content": "Error: partial"})
        follow = msgs[3]
        self.assertEqual(follow["role"], "user")
        kinds = [p["type"] for p in follow["content"]]
        self.assertEqual(kinds, ["text", "image_url", "text"])  # tool media merged with the user's text
        self.assertEqual(follow["content"][-1]["text"], "continue")

    def test_thinking_echoed_as_reasoning_content(self):
        req = _req(messages=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hmm", "signature": ""},
                {"type": "text", "text": "a"},
            ]},
            {"role": "user", "content": "next"},
        ])
        sanitized, _ = prepare_anthropic_adapter_request(req, transport="chat", allow_thinking=True, echo_reasoning=True)
        msgs = _messages_dump(self.c.convert_request(sanitized, "m"))
        self.assertEqual(msgs[1]["reasoning_content"], "hmm")
        # Without echo the block is dropped (strict OpenAI/Azure/Google).
        sanitized, meta = prepare_anthropic_adapter_request(req, transport="chat", allow_thinking=True)
        msgs = _messages_dump(AnthropicToOpenAIConverter().convert_request(sanitized, "m"))
        self.assertNotIn("reasoning_content", msgs[1])
        self.assertIn("messages.thinking", meta.dropped_fields)

    def test_reasoning_effort_mapping(self):
        self.assertEqual(reasoning_effort_for({"type": "enabled", "budget_tokens": 1024}, None), "low")
        self.assertEqual(reasoning_effort_for({"type": "enabled", "budget_tokens": 3000}, None), "medium")
        self.assertEqual(reasoning_effort_for({"type": "enabled", "budget_tokens": 10000}, None), "high")
        self.assertEqual(reasoning_effort_for({"type": "adaptive"}, None), "medium")
        self.assertEqual(reasoning_effort_for({"type": "adaptive"}, {"effort": "xhigh"}), "high")
        self.assertIsNone(reasoning_effort_for({"type": "disabled"}, {"effort": "low"}))
        self.assertEqual(reasoning_effort_for(None, {"effort": "low"}), "low")
        out = self.c.convert_request(_req(thinking={"type": "enabled", "budget_tokens": 5000}), "m")
        self.assertEqual(out.reasoning_effort, "high")

    def test_top_k_goes_in_extra_body(self):
        out = self.c.convert_request(_req(top_k=40), "m").model_dump(exclude_none=True)
        self.assertEqual(out["extra_body"], {"top_k": 40})
        self.assertNotIn("top_k", out)
        out = AnthropicToOpenAIConverter(supports_top_k=False).convert_request(_req(top_k=40), "m")
        self.assertNotIn("extra_body", out.model_dump(exclude_none=True))

    def test_tool_options(self):
        req = _req(
            tools=[{"name": "f", "strict": True, "input_schema": {"properties": {"a": {"type": "string"}}}}],
            tool_choice={"type": "any", "disable_parallel_tool_use": True},
            output_config={"format": {"type": "json_schema", "schema": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}}},
        )
        out = self.c.convert_request(req, "m").model_dump(exclude_none=True)
        fn = out["tools"][0]["function"]
        self.assertTrue(fn["strict"])
        self.assertEqual(fn["parameters"]["type"], "object")
        self.assertEqual(out["tool_choice"], "required")
        self.assertIs(out["parallel_tool_calls"], False)
        schema = out["response_format"]["json_schema"]["schema"]
        self.assertEqual(out["response_format"]["type"], "json_schema")
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(schema["required"], ["x"])
        self.assertIs(out["response_format"]["json_schema"]["strict"], True)

    def test_optional_properties_disable_strict_schema(self):
        # Strict mode would force optional fields to be produced; keep the
        # client's schema as-is and send it non-strict instead.
        for schema in (
            {"type": "object", "properties": {"x": {"type": "integer"}, "notes": {"type": "string"}}, "required": ["x"]},
            {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"], "additionalProperties": True},
        ):
            out = self.c.convert_request(_req(output_config={"format": {"type": "json_schema", "schema": schema}}), "m")
            fmt = out.model_dump(exclude_none=True)["response_format"]["json_schema"]
            self.assertIs(fmt["strict"], False)
            self.assertEqual(fmt["schema"], schema)

    def test_forced_tool_relaxed_when_it_was_dropped(self):
        req = _req(
            tools=[{"type": "web_search_20250305", "name": "web_search"}, {"name": "f", "input_schema": {"type": "object"}}],
            tool_choice={"type": "tool", "name": "web_search"},
        )
        out = self.c.convert_request(req, "m")
        self.assertEqual(out.tool_choice, "auto")

    def test_long_and_invalid_tool_names_round_trip(self):
        long_name = "mcp__" + "very_long_server_name_" * 4 + "tool"
        req = _req(
            messages=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "ns.tool:v1", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
            ],
            tools=[{"name": long_name, "input_schema": {"type": "object"}}, {"name": "ns.tool:v1", "input_schema": {"type": "object"}}],
            tool_choice={"type": "tool", "name": long_name},
        )
        out = self.c.convert_request(req, "m").model_dump(exclude_none=True)
        names = [t["function"]["name"] for t in out["tools"]]
        self.assertTrue(all(len(n) <= 64 and n.replace("_", "").replace("-", "").isalnum() for n in names))
        self.assertEqual(out["tool_choice"]["function"]["name"], names[0])
        self.assertEqual(out["messages"][1]["tool_calls"][0]["function"]["name"], names[1])

        back = OpenAIToAnthropicConverter(tool_names=self.c.tool_names).convert_response({
            "choices": [{"message": {"tool_calls": [{"id": "c1", "function": {"name": names[0], "arguments": "{}"}}]}, "finish_reason": "tool_calls"}],
        }, "m")
        self.assertEqual(back["content"][0]["name"], long_name)

    def test_billing_header_stripped_from_system(self):
        req = _req(system=[{"type": "text", "text": "x-anthropic-billing-header: cc_version=1; cch=abc;\nYou are Claude Code."}])
        out = self.c.convert_request(req, "m")
        self.assertEqual(_messages_dump(out)[0], {"role": "system", "content": "You are Claude Code."})

    def test_foreign_server_tool_history_is_flattened(self):
        req = _req(messages=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [
                {"type": "server_tool_use", "id": "srvtoolu_01abc", "name": "web_search", "input": {"query": "py"}},
                {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_01abc", "content": [
                    {"type": "web_search_result", "url": "https://python.org", "title": "Python", "encrypted_content": "x"},
                ]},
                {"type": "text", "text": "Python is great."},
            ]},
            {"role": "user", "content": "more"},
        ])
        sanitized, _ = prepare_anthropic_adapter_request(req, transport="chat")
        msgs = _messages_dump(self.c.convert_request(sanitized, "m"))
        self.assertEqual(msgs[1]["role"], "assistant")
        self.assertIn("[web_search: py]", msgs[1]["content"])
        self.assertIn("https://python.org", msgs[1]["content"])
        self.assertNotIn("tool_calls", msgs[1])

    def test_prepare_strips_empty_text_and_sanitizes_ids(self):
        req = _req(messages=[
            {"role": "user", "content": [{"type": "text", "text": "  "}, {"type": "text", "text": "real"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "functions.Bash:0", "name": "Bash", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "functions.Bash:0", "content": "ok"}]},
        ], stop_sequences=["a", "b", "c", "d", "e"])
        sanitized, meta = prepare_anthropic_adapter_request(req, transport="chat")
        dumped = sanitized.model_dump(exclude_none=True)
        self.assertEqual(len(dumped["messages"][0]["content"]), 1)
        tool_id = dumped["messages"][1]["content"][0]["id"]
        self.assertRegex(tool_id, r"^[a-zA-Z0-9_-]+$")
        self.assertEqual(dumped["messages"][2]["content"][0]["tool_use_id"], tool_id)
        # The upstream gets its own id back, byte for byte.
        msgs = _messages_dump(AnthropicToOpenAIConverter().convert_request(sanitized, "m"))
        self.assertEqual(msgs[1]["tool_calls"][0]["id"], "functions.Bash:0")
        self.assertEqual(msgs[2]["tool_call_id"], "functions.Bash:0")
        self.assertEqual(dumped["stop_sequences"], ["a", "b", "c", "d"])
        self.assertIn("stop_sequences", meta.dropped_fields)

    def test_responses_transport_options(self):
        req = _req(
            tools=[{"name": "f", "input_schema": {"type": "object"}}],
            tool_choice={"type": "tool", "name": "f", "disable_parallel_tool_use": True},
            thinking={"type": "enabled", "budget_tokens": 2048},
            messages=[{"role": "user", "content": [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBE"}},
            ]}],
        )
        out = AnthropicToResponsesConverter().convert_request(req, "codex").model_dump(exclude_none=True)
        self.assertEqual(out["tool_choice"], {"type": "function", "name": "f"})
        self.assertIs(out["parallel_tool_calls"], False)
        self.assertEqual(out["reasoning"], {"effort": "medium", "summary": "auto"})
        self.assertEqual(out["input"][0]["content"][0]["type"], "input_file")


# ---------------------------------------------------------------------------
# Response conversion
# ---------------------------------------------------------------------------

class ResponseConversionTests(unittest.TestCase):
    def _resp(self, message, finish_reason="stop", usage=None, **choice_extra):
        return {
            "choices": [{"message": message, "finish_reason": finish_reason, **choice_extra}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 2},
        }

    def test_tool_calls_with_stop_finish_reason_are_tool_use(self):
        out = OpenAIToAnthropicConverter().convert_response(self._resp(
            {"content": None, "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": "{}"}}]},
            finish_reason="stop",
        ), "m")
        self.assertEqual(out["stop_reason"], "tool_use")

    def test_matched_stop_sequence_reported(self):
        conv = OpenAIToAnthropicConverter(stop_sequences=["END"])
        out = conv.convert_response(self._resp({"content": "x"}, stop_reason="END"), "m")
        self.assertEqual((out["stop_reason"], out["stop_sequence"]), ("stop_sequence", "END"))
        out = conv.convert_response(self._resp({"content": "x"}, stop_reason=None), "m")
        self.assertEqual((out["stop_reason"], out["stop_sequence"]), ("end_turn", None))

    def test_cached_usage_mapped(self):
        out = OpenAIToAnthropicConverter().convert_response(self._resp(
            {"content": "x"}, usage={"prompt_tokens": 100, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 60}},
        ), "m")
        self.assertEqual(out["usage"], {"input_tokens": 40, "output_tokens": 5, "cache_read_input_tokens": 60})

    def test_bad_arguments_repaired_or_preserved(self):
        self.assertEqual(parse_tool_arguments('{"a": 1'), {"a": 1})
        self.assertEqual(parse_tool_arguments('{"a": "x'), {"a": "x"})
        self.assertEqual(parse_tool_arguments("not json"), {"_raw_arguments": "not json"})
        self.assertEqual(parse_tool_arguments("[1,2]"), {"_raw_arguments": "[1,2]"})
        self.assertEqual(parse_tool_arguments(""), {})

    def test_missing_tool_call_id_tolerated(self):
        validated = ChatCompletionResponse(**{
            "id": "r", "created": 1, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }).model_dump(exclude_unset=True)
        out = OpenAIToAnthropicConverter().convert_response(validated, "m")
        self.assertTrue(out["content"][0]["id"].startswith("toolu_"))

    def test_url_citation_annotations_become_citations(self):
        text = "See Python docs for details."
        out = OpenAIToAnthropicConverter().convert_response(self._resp({
            "content": text,
            "annotations": [{"type": "url_citation", "url_citation": {"url": "https://docs.python.org", "title": "Docs", "start_index": 4, "end_index": 15}}],
        }), "m")
        citation = out["content"][0]["citations"][0]
        self.assertEqual(citation["type"], "web_search_result_location")
        self.assertEqual(citation["cited_text"], "Python docs")

    def test_think_tags_split_when_enabled(self):
        out = OpenAIToAnthropicConverter(split_think_tags=True).convert_response(
            self._resp({"content": "<think>reason</think>\n\nanswer"}), "m")
        self.assertEqual(out["content"][0], {"type": "thinking", "thinking": "reason", "signature": ""})
        self.assertEqual(out["content"][1], {"type": "text", "text": "answer"})


# ---------------------------------------------------------------------------
# Streaming conversion
# ---------------------------------------------------------------------------

class StreamConversionTests(unittest.TestCase):
    def _run(self, chunks, converter=None):
        conv = converter or OpenAIToAnthropicConverter()
        state = conv.new_stream_state("msg_1", "m")
        out = []
        for chunk in chunks:
            out.extend(conv.convert_stream_chunk(chunk, state))
        out.extend(conv.convert_stream_end(state))
        return _parse_sse(out)

    @staticmethod
    def _assert_well_formed(test, events):
        open_index = None
        closed = set()
        for etype, data in events:
            if etype == "content_block_start":
                test.assertIsNone(open_index, "a block was started while another is open")
                open_index = data["index"]
            elif etype == "content_block_delta":
                test.assertEqual(data["index"], open_index, "delta for a block that is not open")
                test.assertNotIn(data["index"], closed)
            elif etype == "content_block_stop":
                test.assertEqual(data["index"], open_index)
                closed.add(open_index)
                open_index = None
        test.assertIsNone(open_index)

    def _tool_inputs(self, events):
        blocks = {}
        for etype, data in events:
            if etype == "content_block_start" and data["content_block"]["type"] == "tool_use":
                blocks[data["index"]] = [data["content_block"]["name"], ""]
            elif etype == "content_block_delta" and data["delta"]["type"] == "input_json_delta":
                blocks[data["index"]][1] += data["delta"]["partial_json"]
        return [(name, json.loads(args)) for name, args in blocks.values()]

    def test_interleaved_parallel_tool_calls(self):
        def tc(index, **kw):
            return {"choices": [{"delta": {"tool_calls": [{"index": index, **kw}]}}]}
        chunks = [
            tc(0, id="a", function={"name": "f", "arguments": '{"x"'}),
            tc(1, id="b", function={"name": "g", "arguments": '{"y"'}),
            tc(0, function={"arguments": ": 1}"}),
            tc(1, function={"arguments": ": 2}"}),
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        events = self._run(chunks)
        self._assert_well_formed(self, events)
        self.assertEqual(self._tool_inputs(events), [("f", {"x": 1}), ("g", {"y": 2})])
        delta = [d for e, d in events if e == "message_delta"][0]
        self.assertEqual(delta["delta"]["stop_reason"], "tool_use")

    def test_parallel_tool_calls_sharing_an_index_stay_separate(self):
        # Gemini's OpenAI layer sends each parallel call whole, all at index 0
        # (or with no index); only the id tells them apart.
        for extra in ({"index": 0}, {}):
            chunks = [
                {"choices": [{"delta": {"tool_calls": [{**extra, "id": "a", "function": {"name": "Bash", "arguments": '{"command": "git diff"}'}}]}}]},
                {"choices": [{"delta": {"tool_calls": [{**extra, "id": "b", "function": {"name": "Read", "arguments": '{"file_path": "/x"}'}}]}}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ]
            events = self._run(chunks)
            self._assert_well_formed(self, events)
            self.assertEqual(self._tool_inputs(events), [("Bash", {"command": "git diff"}), ("Read", {"file_path": "/x"})])
            ids = [d["content_block"]["id"] for e, d in events if e == "content_block_start"]
            self.assertEqual(ids, ["a", "b"])

    def test_streamed_thought_signature_rides_in_tool_id(self):
        chunks = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "f", "arguments": "{}"},
                                                    "extra_content": {"google": {"thought_signature": "sig/1+="}}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c2", "function": {"name": "f", "arguments": "{}"}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        events = self._run(chunks)
        self._assert_well_formed(self, events)
        ids = [d["content_block"]["id"] for e, d in events if e == "content_block_start"]
        self.assertEqual([restore_tool_id(i) for i in ids], ["c1", "c2"])
        self.assertEqual([tool_id_signature(i) for i in ids], ["sig/1+=", None])

    def test_live_tool_call_then_second_call_at_same_index(self):
        def tc(**kw):
            return {"choices": [{"delta": {"tool_calls": [{"index": 0, **kw}]}}]}
        chunks = [
            tc(id="a", function={"name": "Bash", "arguments": '{"comm'}),
            tc(function={"arguments": 'and": "ls"}'}),
            tc(id="b", function={"name": "Read", "arguments": '{"file_path"'}),
            tc(function={"arguments": ': "/x"}'}),
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        events = self._run(chunks)
        self._assert_well_formed(self, events)
        self.assertEqual(self._tool_inputs(events), [("Bash", {"command": "ls"}), ("Read", {"file_path": "/x"})])

    def test_tool_call_ids_repeated_or_regenerated_per_chunk_do_not_split(self):
        # Some upstreams repeat the id on every delta; others send a fresh id
        # on each delta but the name only on the first.
        for second_id in ("a", "a2"):
            chunks = [
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "a", "function": {"name": "Bash", "arguments": '{"comm'}}]}}]},
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": second_id, "function": {"arguments": 'and": "ls"}'}}]}}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ]
            events = self._run(chunks)
            self._assert_well_formed(self, events)
            self.assertEqual(self._tool_inputs(events), [("Bash", {"command": "ls"})])
            ids = [d["content_block"]["id"] for e, d in events if e == "content_block_start"]
            self.assertEqual(ids, ["a"])

    def test_resent_tool_call_id_reuses_its_call(self):
        def tc(**kw):
            return {"choices": [{"delta": {"tool_calls": [{"index": 0, **kw}]}}]}
        chunks = [
            tc(id="a", function={"name": "Bash", "arguments": '{"command"'}),
            tc(id="b", function={"name": "Read", "arguments": '{"file_path": "/x"}'}),
            tc(id="a", function={"arguments": ': "ls"}'}),
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        events = self._run(chunks)
        self._assert_well_formed(self, events)
        self.assertEqual(self._tool_inputs(events), [("Bash", {"command": "ls"}), ("Read", {"file_path": "/x"})])
        ids = [d["content_block"]["id"] for e, d in events if e == "content_block_start"]
        self.assertEqual(ids, ["a", "b"])

    def test_null_and_missing_tool_call_index_are_the_same_call(self):
        chunks = [
            {"choices": [{"delta": {"tool_calls": [{"index": None, "id": "a", "function": {"name": "Bash", "arguments": '{"comm'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": 'and": "ls"}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        events = self._run(chunks)
        self._assert_well_formed(self, events)
        self.assertEqual(self._tool_inputs(events), [("Bash", {"command": "ls"})])

    def test_text_after_tool_call_is_deferred(self):
        chunks = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "a", "function": {"name": "f", "arguments": "{}"}}]}}]},
            {"choices": [{"delta": {"content": "done"}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        events = self._run(chunks)
        self._assert_well_formed(self, events)
        starts = [d["content_block"]["type"] for e, d in events if e == "content_block_start"]
        self.assertEqual(starts, ["tool_use", "text"])

    def test_thinking_block_closes_with_signature_delta(self):
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "think"}}]},
            {"choices": [{"delta": {"content": "answer"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 9, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 4}}},
        ]
        events = self._run(chunks)
        self._assert_well_formed(self, events)
        types = [(e, (d.get("delta") or {}).get("type")) for e, d in events]
        sig_pos = types.index(("content_block_delta", "signature_delta"))
        self.assertEqual(types[sig_pos + 1][0], "content_block_stop")
        delta = [d for e, d in events if e == "message_delta"][0]
        self.assertEqual(delta["usage"], {"input_tokens": 5, "output_tokens": 3, "cache_read_input_tokens": 4})

    def test_think_tag_splitter_across_chunks(self):
        splitter = ThinkTagSplitter()
        parts = []
        for piece in ["<th", "ink>rea", "son</thi", "nk>\n\nans", "wer"]:
            parts.extend(splitter.feed(piece))
        parts.extend(splitter.flush())
        thinking = "".join(t for k, t in parts if k == "thinking")
        text = "".join(t for k, t in parts if k == "text")
        self.assertEqual((thinking, text), ("reason", "answer"))

    def test_stream_annotations_become_citations_delta(self):
        chunks = [
            {"choices": [{"delta": {"content": "Python docs"}}]},
            {"choices": [{"delta": {"annotations": [{"type": "url_citation", "url_citation": {"url": "https://docs.python.org", "title": "D", "start_index": 0, "end_index": 6}}]}, "finish_reason": "stop"}]},
        ]
        events = self._run(chunks)
        self._assert_well_formed(self, events)
        citations = [d["delta"]["citation"] for e, d in events if e == "content_block_delta" and d["delta"]["type"] == "citations_delta"]
        self.assertEqual(citations[0]["cited_text"], "Python")

    def test_matched_stop_sequence_in_stream(self):
        conv = OpenAIToAnthropicConverter(stop_sequences=["STOP"])
        events = self._run([
            {"choices": [{"delta": {"content": "x"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop", "stop_reason": "STOP"}]},
        ], conv)
        delta = [d for e, d in events if e == "message_delta"][0]["delta"]
        self.assertEqual(delta, {"stop_reason": "stop_sequence", "stop_sequence": "STOP"})


# ---------------------------------------------------------------------------
# Adapter execution: errors and rejected-parameter retry
# ---------------------------------------------------------------------------

def _custom_provider(name="t", supported_apis=None, **extra):
    config = {
        "provider_type": "custom",
        "provider_name": "custom",
        "name": name,
        "base_url": "http://upstream.test/v1",
        "api_key": "k",
        "supported_apis": supported_apis or ["openai"],
    }
    config.update(extra)
    return create_custom_provider(config)


def _chat_response(message, finish_reason="stop"):
    return ChatCompletionResponse(**{
        "id": "r", "created": 1, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    })


def _openai_error(status, message):
    return ProviderHTTPError(status, message, body={"error": {"message": message, "type": "invalid_request_error", "code": status}})


class AdapterExecutionTests(unittest.TestCase):
    def setUp(self):
        anthropic_adapter._rejected_params.clear()
        self.addCleanup(anthropic_adapter._rejected_params.clear)

    def test_rejected_param_is_retried_and_remembered(self):
        provider = _custom_provider()
        calls = []

        async def chat_completion(request):
            calls.append(request.model_dump(exclude_none=True))
            if "reasoning_effort" in calls[-1]:
                raise _openai_error(400, "Unrecognized request argument supplied: reasoning_effort")
            return _chat_response({"content": "ok"})

        provider.chat_completion = chat_completion
        req = _req(thinking={"type": "enabled", "budget_tokens": 2048})
        out = run(anthropic_adapter_messages(provider, req, transport="chat", allow_thinking=True))
        self.assertEqual(out["content"], [{"type": "text", "text": "ok"}])
        self.assertEqual(len(calls), 2)
        # Remembered: the next request skips the param up front and reports it.
        run(anthropic_adapter_messages(provider, req, transport="chat", allow_thinking=True))
        self.assertEqual(len(calls), 3)
        self.assertNotIn("reasoning_effort", calls[2])
        self.assertIn("reasoning_effort", provider.get_anthropic_request_metadata(req).dropped_fields)

    def test_rejected_reasoning_content_is_stripped_and_retried(self):
        provider = _custom_provider()
        calls = []

        async def chat_completion(request):
            calls.append(request.model_dump(exclude_none=True))
            if any("reasoning_content" in m for m in calls[-1]["messages"]):
                raise _openai_error(400, "'messages.1' : for 'role:assistant' the following must be satisfied[('messages.1' : property 'reasoning_content' is unsupported)]")
            return _chat_response({"content": "ok"})

        provider.chat_completion = chat_completion
        req = _req(messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm", "signature": "s"}, {"type": "text", "text": "yo"}]},
            {"role": "user", "content": "again"},
        ])
        out = run(provider.anthropic_messages(req))
        self.assertEqual(out["content"], [{"type": "text", "text": "ok"}])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["messages"][1]["content"], "yo")
        run(provider.anthropic_messages(req))
        self.assertEqual(len(calls), 3)
        self.assertIn("reasoning_content", provider.get_anthropic_request_metadata(req).dropped_fields)

    def test_param_named_only_in_prose_is_not_retried(self):
        provider = _custom_provider()
        calls = []

        async def chat_completion(request):
            calls.append(request.model_dump(exclude_none=True))
            raise _openai_error(400, "Conversation roles must alternate user/assistant/user/assistant/...")

        provider.chat_completion = chat_completion
        req = _req(metadata={"user_id": "u1"}, stop_sequences=["x"])
        with self.assertRaises(ProviderHTTPError):
            run(anthropic_adapter_messages(provider, req, transport="chat"))
        self.assertEqual(len(calls), 1)
        self.assertEqual(anthropic_adapter.known_rejected_params(provider, provider.get_model_id(req.model), "chat"), set())

    def test_rejections_reset_when_provider_is_rebuilt(self):
        provider = _custom_provider()
        anthropic_adapter._remember_rejected(provider, "m", "chat", "top_k")
        self.assertEqual(anthropic_adapter.known_rejected_params(provider, "m", "chat"), {"top_k"})
        self.assertEqual(anthropic_adapter.known_rejected_params(provider, "m", "responses"), set())
        self.assertEqual(anthropic_adapter.known_rejected_params(_custom_provider(), "m", "chat"), set())

    def test_errors_are_anthropic_shaped(self):
        provider = _custom_provider()

        async def chat_completion(request):
            raise _openai_error(429, "slow down")

        provider.chat_completion = chat_completion
        with self.assertRaises(ProviderHTTPError) as ctx:
            run(anthropic_adapter_messages(provider, _req(), transport="chat"))
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(ctx.exception.body, {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}})

    def _stream(self, provider, req, **kw):
        async def collect():
            return [c async for c in anthropic_adapter_messages_stream(provider, req, transport="chat", **kw)]
        return _parse_sse(run(collect()))

    def test_stream_retry_happens_before_message_start(self):
        provider = _custom_provider()
        attempts = []

        async def chat_completion_stream(request):
            attempts.append(request.model_dump(exclude_none=True))
            if "extra_body" in attempts[-1]:
                yield 'data: {"error": {"message": "top_k is not supported", "type": "invalid_request_error", "code": 400}}\n\n'
                yield "data: [DONE]\n\n"
                return
            yield 'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}\n\n'
            yield "data: [DONE]\n\n"

        provider.chat_completion_stream = chat_completion_stream
        events = self._stream(provider, _req(top_k=5, stream=True))
        self.assertEqual(len(attempts), 2)
        self.assertEqual([e for e, _ in events].count("message_start"), 1)
        self.assertNotIn("error", [e for e, _ in events])
        self.assertIn(("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}), events)

    def test_stream_error_is_typed(self):
        provider = _custom_provider()

        async def chat_completion_stream(request):
            yield 'data: {"error": {"message": "nope", "type": "x", "code": 401}}\n\n'

        provider.chat_completion_stream = chat_completion_stream
        events = self._stream(provider, _req(stream=True))
        self.assertEqual(events[0][0], "message_start")
        self.assertEqual(events[-1], ("error", {"type": "error", "error": {"type": "authentication_error", "message": "nope"}}))

    def test_message_start_carries_input_estimate(self):
        provider = _custom_provider()

        async def chat_completion_stream(request):
            yield 'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}\n\n'

        provider.chat_completion_stream = chat_completion_stream
        events = self._stream(provider, _req(stream=True, messages=[{"role": "user", "content": "hello there " * 20}]))
        self.assertGreater(events[0][1]["message"]["usage"]["input_tokens"], 20)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class CustomProviderRoutingTests(unittest.TestCase):
    def test_openai_only_provider_uses_adapter(self):
        provider = _custom_provider()
        self.assertEqual(provider.get_anthropic_mode_for_model("custom:t/llama3"), "adapter")
        self.assertIn("anthropic", provider.get_supported_apis_for_model("custom:t/llama3"))
        self.assertEqual(provider.get_anthropic_mode_for_model("custom:t/nomic-embed-text"), "unsupported")
        self.assertNotIn("anthropic", provider.get_supported_apis_for_model("custom:t/bge-m3"))
        self.assertEqual(provider.get_anthropic_request_metadata(_req()).mode, "adapter")
        self.assertIsNone(provider._anthropic_client)

    def test_native_provider_keeps_passthrough(self):
        provider = _custom_provider(supported_apis=["openai", "anthropic"])
        self.assertEqual(provider.get_anthropic_mode_for_model("custom:t/deepseek-chat"), "native")
        self.assertEqual(str(provider._anthropic_client.base_url).rstrip("/"), "http://upstream.test")

    def test_anthropic_base_url_and_bearer_auth(self):
        provider = _custom_provider(
            supported_apis=["openai", "anthropic"],
            anthropic_base_url="https://api.deepseek.com/anthropic",
            anthropic_auth_style="bearer",
        )
        client = provider._anthropic_client
        self.assertEqual(str(client.base_url), "https://api.deepseek.com/anthropic/")
        self.assertEqual(client.auth_headers, {"Authorization": "Bearer k"})

    def test_native_client_ignores_proxy_anthropic_env_credentials(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "proxy-key", "ANTHROPIC_AUTH_TOKEN": "proxy-token"}):
            bearer = _custom_provider(supported_apis=["openai", "anthropic"], anthropic_auth_style="bearer")
            api_key = _custom_provider(supported_apis=["openai", "anthropic"])
        self.assertEqual(bearer._anthropic_client.auth_headers, {"Authorization": "Bearer k"})
        self.assertEqual(api_key._anthropic_client.auth_headers, {"X-Api-Key": "k"})

    def test_adapter_messages_end_to_end(self):
        provider = _custom_provider()

        async def chat_completion(request):
            return _chat_response({"content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}]}, "tool_calls")

        provider.chat_completion = chat_completion
        out = run(provider.anthropic_messages(_req(tools=[{"name": "f", "input_schema": {"type": "object"}}])))
        self.assertEqual(out["stop_reason"], "tool_use")
        self.assertEqual(out["content"][0], {"type": "tool_use", "id": "call_1", "name": "f", "input": {"a": 1}})


class AzureFoundryRoutingTests(unittest.TestCase):
    def _provider(self):
        from app.providers.azure_provider import AzureProvider
        return AzureProvider({
            "name": "foundry",
            "endpoint": "https://example.openai.azure.com/",
            "api_key": "k",
            "azure_backend": "foundry",
            "dynamic_discovery": False,
            "openai_deployments": ["grok-3"],
            "anthropic_deployments": ["claude-sonnet-4"],
            "deployments": ["grok-3", "claude-sonnet-4"],
        })

    def test_non_anthropic_deployments_are_served_through_the_adapter(self):
        provider = self._provider()
        self.assertEqual(provider.get_anthropic_mode_for_model("azure:foundry/grok-3"), "adapter")
        self.assertIn("anthropic", provider.get_supported_apis_for_model("azure:foundry/grok-3"))
        self.assertEqual(provider.get_anthropic_mode_for_model("azure:foundry/claude-sonnet-4"), "native")
        self.assertEqual(provider.get_supported_apis_for_model("azure:foundry/claude-sonnet-4"), ["anthropic"])


    def test_foundry_codex_deployment_uses_responses_transport(self):
        provider = self._provider()
        self.assertEqual(provider._get_adapter_transport("azure:foundry/gpt-5-codex"), "responses")
        self.assertEqual(provider._get_adapter_transport("azure:foundry/grok-3"), "chat")
        self.assertFalse(provider.anthropic_adapter_supports_file_parts)


# ---------------------------------------------------------------------------
# Web search through the real adapter path
# ---------------------------------------------------------------------------

WS_CFG = WebSearchConfig(
    enabled=True,
    searxng_base_url="http://searx.test",
    max_results=3,
    max_agentic_loops=3,
    enabled_providers=frozenset({"custom:t"}),
)


class AdapterWebSearchTests(unittest.TestCase):
    def setUp(self):
        self._saved = websearch_settings_cache.config
        websearch_settings_cache.set_config(WS_CFG)
        self.addCleanup(lambda: websearch_settings_cache.set_config(self._saved))
        self.searches = []

        async def fake_run_searches(queries, cfg):
            self.searches.append((list(queries), cfg))
            return [SearchSucceeded(q, [SearchResult("Py", "https://python.org/news", "Python 3.14 released")]) for q in queries]

        patcher = patch("app.websearch.loop.run_searches", fake_run_searches)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.provider = _custom_provider()
        self.chat_requests = []

    def _script(self, responses):
        async def chat_completion(request):
            self.chat_requests.append(request.model_dump(exclude_none=True))
            return responses.pop(0)
        self.provider.chat_completion = chat_completion

    def _ws_req(self, **extra):
        body = {"tools": [{"type": "web_search_20250305", "name": "web_search"}], "messages": [{"role": "user", "content": "python news?"}]}
        body.update(extra)
        return _req(**body)

    def test_search_tool_reaches_model_and_loop_runs(self):
        self._script([
            _chat_response({"content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "web_search_proxy", "arguments": '{"query": "python news"}'}}]}, "tool_calls"),
            _chat_response({"content": "Python 3.14 is out, see https://python.org/news."}),
        ])
        loop = maybe_intercept_anthropic(self._ws_req(), self.provider)
        result = run(loop.run())
        # The internal tool was sent upstream as a function (regression: it was dropped).
        self.assertEqual(self.chat_requests[0]["tools"][0]["function"]["name"], "web_search_proxy")
        self.assertEqual(self.searches[0][0], ["python news"])
        types = [b["type"] for b in result["content"]]
        self.assertEqual(types, ["server_tool_use", "web_search_tool_result", "text"])
        citations = result["content"][2]["citations"]
        self.assertEqual(citations[0]["url"], "https://python.org/news")
        self.assertEqual(result["usage"]["server_tool_use"], {"web_search_requests": 1})
        # Second round carries the tool call + result as OpenAI messages.
        self.assertEqual(self.chat_requests[1]["messages"][-1]["role"], "tool")

    def test_streaming_search_through_adapter(self):
        streams = [
            ['data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "web_search_proxy", "arguments": "{\\"query\\": \\"py\\"}"}}]}, "finish_reason": "tool_calls"}]}\n\n'],
            ['data: {"choices": [{"delta": {"content": "See https://python.org/news"}, "finish_reason": "stop"}]}\n\n'],
        ]

        async def chat_completion_stream(request):
            for chunk in streams.pop(0):
                yield chunk
            yield "data: [DONE]\n\n"

        self.provider.chat_completion_stream = chat_completion_stream
        loop = maybe_intercept_anthropic(self._ws_req(stream=True), self.provider)

        async def collect():
            return [c async for c in loop.stream()]
        events = _parse_sse(run(collect()))
        starts = [d["content_block"]["type"] for e, d in events if e == "content_block_start"]
        self.assertEqual(starts, ["server_tool_use", "web_search_tool_result", "text"])
        self.assertTrue(any(e == "content_block_delta" and d["delta"]["type"] == "citations_delta" for e, d in events))
        self.assertEqual([e for e, _ in events][-1], "message_stop")

    def test_tool_options_applied(self):
        loop = maybe_intercept_anthropic(self._ws_req(tools=[{
            "type": "web_search_20250305", "name": "web_search", "max_uses": 1,
            "allowed_domains": ["python.org"], "user_location": {"type": "approximate", "country": "DE"},
        }]), self.provider)
        self.assertEqual(loop.cfg.max_uses, 1)
        self.assertEqual(loop.cfg.allowed_domains, ("python.org",))
        self.assertEqual(loop.cfg.fourget_country, "de")
        outcomes = run(loop._guard.search(["a", "b"]))
        self.assertTrue(outcomes[0].ok)
        self.assertEqual(outcomes[1].error_code, "max_uses_exceeded")
        self.assertEqual(loop.stats.executed, 1)

    def test_both_domain_lists_rejected(self):
        with self.assertRaises(WebSearchRequestError):
            maybe_intercept_anthropic(self._ws_req(tools=[{
                "type": "web_search_20250305", "name": "web_search",
                "allowed_domains": ["a.com"], "blocked_domains": ["b.com"],
            }]), self.provider)

    def test_claude_code_shortcircuit(self):
        self._script([_chat_response({"content": "Summary citing https://python.org/news"})])
        req = self._ws_req(
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}],
            messages=[{"role": "user", "content": [{"type": "text", "text": "Perform a web search for the query: python 3.14 release"}]}],
        )
        loop = maybe_intercept_anthropic(req, self.provider)
        self.assertEqual(loop.shortcircuit_query, "python 3.14 release")
        result = run(loop.run())
        self.assertEqual(len(self.chat_requests), 1)
        self.assertEqual(self.chat_requests[0]["tool_choice"], "none")
        self.assertEqual(self.searches[0][0], ["python 3.14 release"])
        types = [b["type"] for b in result["content"]]
        self.assertEqual(types, ["server_tool_use", "web_search_tool_result", "text"])
        self.assertEqual(result["content"][0]["input"], {"query": "python 3.14 release"})

    def test_replay_strips_proxy_citations(self):
        citation = tools.citations_for_text("https://python.org/news", [SearchResult("Py", "https://python.org/news", "s")])[0]
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "text", "text": "https://python.org/news", "citations": [citation]}]},
            {"role": "user", "content": "more"},
        ]
        self.assertTrue(tools.history_has_replayed_blocks(messages))
        out = tools.rehydrate_anthropic_messages(messages)
        self.assertNotIn("citations", out[1]["content"][0])


class DomainFilterTests(unittest.TestCase):
    def test_allowed_and_blocked_domains_filter_results(self):
        seen = {}

        async def fake_backend(query, cfg):
            seen["query"] = query
            return SearchSucceeded(query, [
                SearchResult("A", "https://docs.python.org/3/", "a"),
                SearchResult("B", "https://evil.example/x", "b"),
            ])

        with patch("app.websearch.backends._backend_search", fake_backend):
            cfg = WS_CFG.with_overrides(allowed_domains=("python.org",))
            outcome = run(backends.search("asyncio", cfg))
            self.assertEqual(seen["query"], "asyncio site:python.org")
            self.assertEqual([r.url for r in outcome.results], ["https://docs.python.org/3/"])
            self.assertEqual(outcome.query, "asyncio")

            cfg = WS_CFG.with_overrides(blocked_domains=("evil.example",))
            outcome = run(backends.search("asyncio", cfg))
            self.assertEqual(seen["query"], "asyncio -site:evil.example")
            self.assertEqual([r.url for r in outcome.results], ["https://docs.python.org/3/"])


class TokenEstimateTests(unittest.TestCase):
    def test_images_are_not_tokenized_as_base64(self):
        small = estimate_anthropic_input_tokens({"messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "A" * 400_000}},
        ]}]})
        self.assertLess(small, 2000)

    def test_tools_and_system_counted(self):
        base = estimate_anthropic_input_tokens({"messages": [{"role": "user", "content": "hi"}]})
        more = estimate_anthropic_input_tokens({
            "system": "You are a helpful assistant with many rules.",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "f", "description": "does things", "input_schema": {"type": "object"}}],
        })
        self.assertGreater(more, base)


def _tool(name, deferred=False):
    tool = {"name": name, "description": name, "input_schema": {"type": "object", "properties": {}}}
    if deferred:
        tool["defer_loading"] = True
    return tool


class DeferLoadingTests(unittest.TestCase):
    """The adapter emulates tool search: deferred tools appear once loaded."""

    def _prepare(self, tools, messages=None):
        req = _req(tools=tools, messages=messages or [{"role": "user", "content": "hi"}])
        sanitized, metadata = prepare_anthropic_adapter_request(req, transport="chat")
        dumped = sanitized.model_dump(exclude_none=True)
        return dumped.get("tools", []), metadata

    def test_only_loaded_deferred_tools_are_declared(self):
        tools, metadata = self._prepare(
            [_tool("ToolSearch"), _tool("Bash"), _tool("A", True), _tool("B", True), _tool("C", True)],
            [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "ToolSearch", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [
                    {"type": "tool_reference", "tool_name": "A"},
                ]}]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "C", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "ok"}]},
            ],
        )
        self.assertEqual([t["name"] for t in tools], ["ToolSearch", "Bash", "A", "C"])
        self.assertFalse(any("defer_loading" in t for t in tools))
        self.assertFalse(any("B" in f for f in metadata.dropped_fields))

    def test_server_tool_search_history_loads_its_references(self):
        tools, _ = self._prepare(
            [_tool("Bash"), _tool("D", True), _tool("E", True)],
            [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [
                    {"type": "server_tool_use", "id": "srvtoolu_1", "name": "tool_search_tool_regex", "input": {"query": "d"}},
                    {"type": "tool_search_tool_result", "tool_use_id": "srvtoolu_1", "content": {
                        "type": "tool_search_tool_search_result",
                        "tool_references": [{"type": "tool_reference", "tool_name": "D"}],
                    }},
                    {"type": "text", "text": "found it"},
                ]},
                {"role": "user", "content": "use it"},
            ],
        )
        self.assertEqual([t["name"] for t in tools], ["Bash", "D"])

    def test_server_side_tool_search_sends_every_tool(self):
        tools, _ = self._prepare([
            {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
            _tool("Bash"), _tool("A", True),
        ])
        self.assertEqual([t["name"] for t in tools], ["Bash", "A"])

    def test_all_deferred_sends_every_tool(self):
        tools, _ = self._prepare([_tool("A", True), _tool("B", True)])
        self.assertEqual([t["name"] for t in tools], ["A", "B"])

    def test_upstream_sees_only_loaded_tools(self):
        provider = _custom_provider()
        calls = []

        async def chat_completion(request):
            calls.append(request.model_dump(exclude_none=True))
            return _chat_response({"content": "ok"})

        provider.chat_completion = chat_completion
        req = _req(tools=[_tool("ToolSearch"), _tool("A", True), _tool("B", True)], messages=[
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "ToolSearch", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [
                {"type": "tool_reference", "tool_name": "B"},
            ]}]},
        ])
        run(anthropic_adapter_messages(provider, req, transport="chat"))
        self.assertEqual([t["function"]["name"] for t in calls[0]["tools"]], ["ToolSearch", "B"])
        self.assertEqual(calls[0]["messages"][-1]["content"], "Tool loaded: B")


class ToolNameMapTests(unittest.TestCase):
    def test_mapping_is_deterministic(self):
        name = "x" * 80
        self.assertEqual(ToolNameMap().to_openai(name), ToolNameMap().to_openai(name))
        self.assertEqual(ToolNameMap().to_openai("ok_name"), "ok_name")


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Responses transport (Anthropic Messages <-> OpenAI Responses)
# ---------------------------------------------------------------------------

def _sse_event(event_type, data, bare=False):
    payload = json.dumps({"type": event_type, **data})
    return f"data: {payload}\n\n" if bare else f"event: {event_type}\ndata: {payload}\n\n"


class ResponsesTransportTests(unittest.TestCase):
    def setUp(self):
        anthropic_adapter._rejected_params.clear()
        self.addCleanup(anthropic_adapter._rejected_params.clear)

    def _convert(self, req):
        sanitized, meta = prepare_anthropic_adapter_request(req, transport="responses", allow_thinking=True)
        return AnthropicToResponsesConverter().convert_request(sanitized, "codex").model_dump(exclude_none=True), meta

    def _stream(self, events):
        conv = ResponsesToAnthropicConverter()
        state = StreamConversionState(message_id="msg_1", model="m")
        out = []
        for event_type, data in events:
            out.extend(conv.convert_stream_event(event_type, data, state))
        out.extend(conv.convert_stream_end(state))
        return _parse_sse(out)

    # -- request --

    def test_max_output_tokens_floor(self):
        out, _ = self._convert(_req(max_tokens=1))
        self.assertEqual(out["max_output_tokens"], 16)

    def test_stop_sequences_and_top_k_reported_dropped(self):
        out, meta = self._convert(_req(stop_sequences=["x"], top_k=5))
        self.assertNotIn("stop", out)
        self.assertIn("stop_sequences", meta.dropped_fields)
        self.assertIn("top_k", meta.dropped_fields)

    def test_function_call_ids_are_stable_across_requests(self):
        req = _req(messages=[
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]},
        ])
        first, _ = self._convert(req)
        second, _ = self._convert(req)
        self.assertEqual(first["input"], second["input"])
        self.assertEqual(first["input"][1]["call_id"], first["input"][2]["call_id"])

    def test_reasoning_summary_requested_for_thinking_only(self):
        out, _ = self._convert(_req(thinking={"type": "adaptive"}))
        self.assertEqual(out["reasoning"], {"effort": "medium", "summary": "auto"})
        out, _ = self._convert(_req(thinking={"type": "enabled", "budget_tokens": 4096, "display": "omitted"}))
        self.assertEqual(out["reasoning"], {"effort": "high"})
        out, _ = self._convert(_req(output_config={"effort": "low"}))
        self.assertEqual(out["reasoning"], {"effort": "low"})

    # -- non-streaming response --

    def _response(self, output, status="completed", **extra):
        return {"id": "resp_1", "status": status, "output": output,
                "usage": {"input_tokens": 10, "output_tokens": 4, "input_tokens_details": {"cached_tokens": 6}}, **extra}

    def test_refusal_and_incomplete_stop_reasons(self):
        conv = ResponsesToAnthropicConverter()
        refusal = self._response([{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}])
        out = conv.convert_response(refusal, "m")
        self.assertEqual(out["stop_reason"], "refusal")
        self.assertEqual(out["content"], [{"type": "text", "text": "no"}])
        self.assertEqual(out["usage"], {"input_tokens": 4, "output_tokens": 4, "cache_read_input_tokens": 6})

        filtered = self._response([], status="incomplete", incomplete_details={"reason": "content_filter"})
        self.assertEqual(conv.convert_response(filtered, "m")["stop_reason"], "refusal")
        truncated = self._response([], status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        self.assertEqual(conv.convert_response(truncated, "m")["stop_reason"], "max_tokens")

    def test_reasoning_items_become_one_thinking_block(self):
        conv = ResponsesToAnthropicConverter()
        out = conv.convert_response(self._response([
            {"type": "reasoning", "id": "rs_1", "summary": [
                {"type": "summary_text", "text": "a"}, {"type": "summary_text", "text": "b"},
            ]},
            {"type": "reasoning", "id": "rs_2", "summary": [], "content": [{"type": "reasoning_text", "text": "raw"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "hi", "annotations": [
                {"type": "url_citation", "url": "https://x.test", "title": "X", "start_index": 0, "end_index": 2},
            ]}]},
        ]), "m")
        self.assertEqual([b["type"] for b in out["content"]], ["thinking", "thinking", "text"])
        self.assertEqual(out["content"][0]["thinking"], "a\n\nb")
        self.assertEqual(out["content"][1]["thinking"], "raw")
        self.assertEqual(out["content"][2]["citations"][0]["url"], "https://x.test")

    def test_failed_response_raises(self):
        provider = _custom_provider()

        async def responses_create(request):
            return {"id": "resp_1", "status": "failed", "output": [],
                    "error": {"code": "rate_limit_exceeded", "message": "slow down"}}

        provider.responses_create = responses_create
        with self.assertRaises(ProviderHTTPError) as ctx:
            run(anthropic_adapter_messages(provider, _req(), transport="responses"))
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(ctx.exception.body["error"]["type"], "rate_limit_error")

    def test_rejected_reasoning_summary_keeps_effort(self):
        provider = _custom_provider()
        calls = []

        async def responses_create(request):
            calls.append(request.model_dump(exclude_none=True))
            if "summary" in (calls[-1].get("reasoning") or {}):
                raise _openai_error(400, "Your organization must be verified to generate reasoning summaries.")
            return {"id": "resp_1", "status": "completed",
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]}

        provider.responses_create = responses_create
        req = _req(thinking={"type": "enabled", "budget_tokens": 4096})
        out = run(anthropic_adapter_messages(provider, req, transport="responses", allow_thinking=True))
        self.assertEqual(out["content"], [{"type": "text", "text": "ok"}])
        self.assertEqual([c["reasoning"] for c in calls], [{"effort": "high", "summary": "auto"}, {"effort": "high"}])

    # -- streaming --

    def test_stream_arguments_only_in_done_event(self):
        events = self._stream([
            ("response.output_item.added", {"output_index": 0, "item": {"type": "function_call", "call_id": "call_1", "name": "f"}}),
            ("response.function_call_arguments.done", {"output_index": 0, "arguments": '{"a": 1}'}),
            ("response.output_item.done", {"output_index": 0, "item": {"type": "function_call", "arguments": '{"a": 1}'}}),
            ("response.completed", {"response": {"status": "completed"}}),
        ])
        StreamConversionTests._assert_well_formed(self, events)
        self.assertEqual(StreamConversionTests._tool_inputs(self, events), [("f", {"a": 1})])
        self.assertEqual(events[-2][1]["delta"]["stop_reason"], "tool_use")

    def test_stream_arguments_not_duplicated_after_deltas(self):
        events = self._stream([
            ("response.output_item.added", {"output_index": 0, "item": {"type": "function_call", "call_id": "call_1", "name": "f"}}),
            ("response.function_call_arguments.delta", {"output_index": 0, "delta": '{"a":'}),
            ("response.function_call_arguments.delta", {"output_index": 0, "delta": ' 1}'}),
            ("response.function_call_arguments.done", {"output_index": 0, "arguments": '{"a": 1}'}),
            ("response.completed", {"response": {"status": "completed"}}),
        ])
        self.assertEqual(StreamConversionTests._tool_inputs(self, events), [("f", {"a": 1})])

    def test_stream_refusal_and_incomplete_usage(self):
        events = self._stream([
            ("response.refusal.delta", {"output_index": 0, "delta": "can't"}),
            ("response.incomplete", {"response": {"status": "incomplete", "incomplete_details": {"reason": "content_filter"},
                                                  "usage": {"input_tokens": 9, "output_tokens": 3}}}),
        ])
        text = "".join(d["delta"]["text"] for t, d in events if t == "content_block_delta")
        self.assertEqual(text, "can't")
        delta = events[-2][1]
        self.assertEqual(delta["delta"]["stop_reason"], "refusal")
        self.assertEqual(delta["usage"], {"input_tokens": 9, "output_tokens": 3})

    def test_stream_summary_parts_are_separated(self):
        events = self._stream([
            ("response.reasoning_summary_part.added", {"summary_index": 0}),
            ("response.reasoning_summary_text.delta", {"delta": "a"}),
            ("response.reasoning_summary_part.added", {"summary_index": 1}),
            ("response.reasoning_summary_text.delta", {"delta": "b"}),
            ("response.completed", {"response": {"status": "completed"}}),
        ])
        thinking = "".join(d["delta"].get("thinking", "") for t, d in events if t == "content_block_delta")
        self.assertEqual(thinking, "a\n\nb")

    def test_stream_failure_ends_with_error_only(self):
        events = self._stream([
            ("response.output_text.delta", {"delta": "par"}),
            ("response.failed", {"response": {"status": "failed", "error": {"code": "server_error", "message": "boom"}}}),
        ])
        self.assertEqual(events[-1], ("error", {"type": "error", "error": {"type": "api_error", "message": "boom"}}))
        self.assertNotIn("message_stop", [t for t, _ in events])
        StreamConversionTests._assert_well_formed(self, events)

    def test_adapter_stream_accepts_bare_data_lines(self):
        provider = _custom_provider()

        async def responses_create_stream(request):
            yield _sse_event("response.created", {"response": {"status": "in_progress"}}, bare=True)
            yield _sse_event("response.output_text.delta", {"delta": "hi"}, bare=True)
            yield _sse_event("response.completed", {"response": {"status": "completed"}}, bare=True)

        provider.responses_create_stream = responses_create_stream
        chunks = run(_collect(anthropic_adapter_messages_stream(provider, _req(stream=True), transport="responses")))
        events = _parse_sse(chunks)
        self.assertEqual([t for t, _ in events][-3:], ["content_block_stop", "message_delta", "message_stop"])
        self.assertIn(("content_block_delta", {"type": "content_block_delta", "index": 0,
                                               "delta": {"type": "text_delta", "text": "hi"}}), events)


async def _collect(stream):
    return [chunk async for chunk in stream]
