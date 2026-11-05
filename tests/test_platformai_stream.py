"""PlatformAI: reply accumulation, SSE parsing and error mapping.

Fixtures are frames captured from the live v3 gateway (ids/signatures shortened).
"""

import asyncio
import json
import unittest

from app.providers.platformai.client import iter_sse, map_error, metadata_header, parse_error
from app.providers.platformai.stream import GatewayTurn, message_text, turns_from_response


def _events(text):
    async def lines():
        for line in text.split("\n"):
            yield line

    async def collect():
        return [e async for e in iter_sse(lines())]

    return asyncio.run(collect())


def _text(t, **extra):
    return {"type": "text", "text": t, **extra}


class StreamAccumulationTests(unittest.TestCase):
    def test_documented_example(self):
        turn = GatewayTurn()
        out = [
            turn.apply_chunk({"model": "m", "messages": [{"role": "assistant", "contents": [_text("Revenue grew")]}]}),
            turn.apply_chunk({"model": "m", "messages": [{"role": "assistant",
                                                          "contents": [_text(" steadily across all three months")]}]}),
            turn.apply_chunk({"id": "chatcmpl-abc123", "model": "m", "created": 1768213531,
                              "usage": {"prompt_tokens": 26, "completion_tokens": 191, "total_tokens": 217},
                              "messages": [{"role": "assistant", "finish_reason": "stop", "contents": [
                                  _text("Revenue grew steadily across all three months.")]}],
                              "is_final_chunk": True}),
        ]
        self.assertEqual("".join(out), "Revenue grew steadily across all three months.")
        self.assertEqual(out[-1], ".")
        self.assertTrue(turn.finished)
        self.assertEqual((turn.id, turn.created, turn.usage["completion_tokens"]), ("chatcmpl-abc123", 1768213531, 191))
        self.assertEqual(turn.finish_reason(), "stop")

    def test_live_text_stream(self):
        frames = [
            {"usage": {"completion_tokens": 2, "prompt_tokens": 16, "total_tokens": 18, "cached_tokens": 0},
             "messages": [{"role": "assistant", "contents": [_text("")]}], "model": "claude-haiku-5-5"},
            {"messages": [{}], "model": "claude-haiku-5-5"},
            {"model": "claude-haiku-5-5"},
            {"messages": [{"contents": [_text("1,")]}], "model": "claude-haiku-5-5"},
            {"messages": [{"contents": [_text(" 2, 3, 4")]}], "model": "claude-haiku-5-5"},
            {"messages": [{"contents": [_text(", 5")]}], "model": "claude-haiku-5-5"},
            {"created": 1793790865, "is_final_chunk": True,
             "usage": {"completion_tokens": 2, "prompt_tokens": 16, "total_tokens": 18},
             "messages": [{"contents": [_text("1, 2, 3, 4, 5")], "index": 0}], "model": "claude-haiku-5-5",
             "id": "1793790865_final", "credit_usage": {"total_credits": 0.0026}},
        ]
        turn = GatewayTurn()
        text = "".join(turn.apply_chunk(f) for f in frames)
        self.assertEqual(text, "1, 2, 3, 4, 5")
        self.assertEqual(turn.finish_reason(), "stop")

    def test_single_consolidated_frame(self):
        # Output guardrails make the gateway send the whole reply as one final frame.
        turn = GatewayTurn()
        self.assertEqual(turn.apply_chunk({"is_final_chunk": True, "messages": [{"contents": [_text("all")]}]}), "all")

    def test_final_text_that_does_not_extend_is_not_re_sent(self):
        turn = GatewayTurn()
        turn.apply_chunk({"messages": [{"contents": [_text("Hello")]}]})
        self.assertEqual(turn.apply_chunk({"is_final_chunk": True, "messages": [{"contents": [_text("Bye")]}]}), "")
        self.assertEqual(turn.text, "Hello")

    def test_reasoning_blocks_are_not_text(self):
        message = {"contents": [_text("", reasoning=True, signature="CAQS..."), _text("Pink")]}
        self.assertEqual(message_text(message), "Pink")
        turn = GatewayTurn()
        self.assertEqual(turn.apply_chunk({"messages": [{"contents": [_text("thinking", reasoning=True)]}]}), "")

    def test_live_tool_call_deltas(self):
        frames = [
            {"messages": [{"role": "assistant", "contents": [_text("")]}], "model": "m"},
            {"messages": [{"contents": [_text("ok let me check")]}], "model": "m"},
            {"messages": [{"tools": [{"function": {"name": "get_weather", "arguments": ""},
                                      "id": "toolu_012e", "type": "function"}]}], "model": "m"},
            {"messages": [{"tools": [{"function": {"arguments": ""}, "type": "function"}]}], "model": "m"},
            {"messages": [{"tools": [{"function": {"arguments": "{\"city\""}, "type": "function"}]}], "model": "m"},
            {"messages": [{"tools": [{"function": {"arguments": ": \"Chen"}, "type": "function"}]}], "model": "m"},
            {"messages": [{"tools": [{"function": {"arguments": "nai\"}"}, "type": "function"}]}], "model": "m"},
        ]
        turn = GatewayTurn()
        self.assertFalse(turn.tool_calls_complete())
        for frame in frames[:5]:
            turn.apply_chunk(frame)
        self.assertFalse(turn.tool_calls_complete())  # '{"city"' is not JSON yet
        for frame in frames[5:]:
            turn.apply_chunk(frame)
        self.assertTrue(turn.tool_calls_complete())
        self.assertEqual(turn.text, "ok let me check")
        self.assertEqual(turn.tool_calls(), [{"index": 0, "id": "toolu_012e", "type": "function",
                                              "function": {"name": "get_weather", "arguments": '{"city": "Chennai"}'}}])
        self.assertEqual(turn.finish_reason(), "tool_calls")

    def test_two_tool_calls_split_by_id(self):
        turn = GatewayTurn()
        turn.apply_chunk({"messages": [{"tools": [{"id": "a", "function": {"name": "f", "arguments": "{}"}}]}]})
        turn.apply_chunk({"messages": [{"tools": [{"id": "b", "function": {"name": "g", "arguments": "{\"x\""}}]}]})
        turn.apply_chunk({"messages": [{"tools": [{"function": {"arguments": ":1}"}}]}]})
        self.assertEqual([(c["id"], c["function"]["arguments"]) for c in turn.tool_calls()],
                         [("a", "{}"), ("b", '{"x":1}')])

    def test_indexed_tool_deltas(self):
        turn = GatewayTurn()
        turn.apply_chunk({"messages": [{"tools": [{"index": 1, "id": "b", "function": {"name": "g", "arguments": "{"}},
                                                  {"index": 0, "id": "a", "function": {"name": "f", "arguments": "{}"}}]}]})
        turn.apply_chunk({"messages": [{"tools": [{"index": 1, "function": {"arguments": "}"}}]}]})
        self.assertEqual([c["id"] for c in turn.tool_calls()], ["a", "b"])

    def test_usage_keeps_largest_counts(self):
        turn = GatewayTurn()
        turn.apply_chunk({"usage": {"completion_tokens": 9, "prompt_tokens": 16}})
        turn.apply_chunk({"is_final_chunk": True, "usage": {"completion_tokens": 2, "prompt_tokens": 16}})
        self.assertEqual(turn.usage["completion_tokens"], 9)

    def test_non_stream_response_and_variations(self):
        turns = turns_from_response({
            "model": "claude-haiku-5-5", "usage": {"prompt_tokens": 1},
            "messages": [
                {"role": "assistant", "index": 1, "finish_reason": "stop", "contents": [_text("second")]},
                {"role": "assistant", "index": 0, "finish_reason": "maxtokens",
                 "contents": [_text("", reasoning=True), _text("first")]},
            ],
        })
        self.assertEqual([t.text for t in turns], ["first", "second"])
        self.assertEqual([t.finish_reason() for t in turns], ["length", "stop"])
        self.assertEqual(turns[0].model, "claude-haiku-5-5")

    def test_non_stream_tool_call(self):
        (turn,) = turns_from_response({"messages": [{"role": "assistant", "finish_reason": "tool_calls", "tools": [
            {"function": {"name": "get_weather", "arguments": "{\"city\":\"Chennai\"}"},
             "id": "toolu_01LK", "type": "function"}]}]})
        self.assertEqual(turn.text, "")
        self.assertEqual(turn.tool_calls()[0]["function"], {"name": "get_weather", "arguments": '{"city":"Chennai"}'})
        self.assertEqual(turn.finish_reason(), "tool_calls")

    def test_empty_response_gives_one_empty_turn(self):
        (turn,) = turns_from_response({})
        self.assertEqual((turn.text, turn.finish_reason()), ("", "stop"))


class SSEParserTests(unittest.TestCase):
    def test_events_and_multiline_data(self):
        events = _events('event: generate\ndata: {"a":\ndata: 1}\n\n: comment\ndata: {"b":2}\n\nevent: ping\ndata: x')
        self.assertEqual([(e.event, e.data) for e in events],
                         [("generate", '{"a":1}'), ("", '{"b":2}'), ("ping", "x")])
        self.assertEqual(json.loads(events[0].data), {"a": 1})


def _err(code, message="m"):
    return json.dumps({"error": {"code": code, "message": message}})


class ErrorMappingTests(unittest.TestCase):
    def test_gateway_codes(self):
        cases = {
            ("PAI-MODEL-002", 400): (400, "invalid_request_error"),
            ("PAI-MODEL-001", 400): (404, "not_found_error"),
            ("PAI-TOKEN-003", 400): (404, "not_found_error"),
            ("PAI-MODEL-007", 400): (403, "permission_error"),
            ("PAI-PROVIDER-001", 400): (429, "rate_limit_error"),
            ("PAI-PROVIDER-002", 400): (400, "invalid_request_error"),
            ("PAI-PROVIDER-005", 400): (503, "upstream_error"),
            ("PAI-OBSERVABILITY-006", 400): (400, "invalid_request_error"),
            ("PAI-TOKEN-001", 400): (502, "upstream_auth_error"),
            ("PAI-WORKSPACE-001", 400): (502, "upstream_config_error"),
            ("API-array_out_of_range", 400): (400, "invalid_request_error"),
            ("API-invalid_oauth_token", 401): (502, "upstream_auth_error"),
            ("API-url_not_found", 404): (502, "upstream_config_error"),
        }
        for (code, status), expected in cases.items():
            with self.subTest(code=code):
                err = map_error(status, _err(code, "boom"))
                self.assertEqual((err.status_code, err.body["error"]["type"]), expected)
                self.assertEqual(err.body["error"]["param"], code)
                self.assertEqual(err.pai_code, code)
                self.assertIn(f"{code}: boom", err.message)

    def test_guardrail_is_content_filter(self):
        err = map_error(424, _err("PAI-GUARDRAIL-001", "blocked"))
        self.assertEqual((err.status_code, err.body["error"]["code"]), (400, "content_filter"))

    def test_status_fallbacks(self):
        self.assertEqual(map_error(401, "nope").status_code, 502)
        self.assertEqual(map_error(429, "slow", None).status_code, 429)
        self.assertEqual(map_error(400, "plain").status_code, 400)
        internal = map_error(500, json.dumps({"error": {"code": 500, "requestURI": "/pai/x",
                                                        "message": "Something went wrong"}}))
        self.assertEqual(internal.status_code, 502)
        self.assertIn("Something went wrong", internal.message)
        self.assertIsNone(internal.pai_code)

    def test_parse_error_shapes(self):
        self.assertEqual(parse_error(_err("PAI-MODEL-002", "x")).code, "PAI-MODEL-002")
        self.assertEqual(parse_error("not json").message, "not json")
        self.assertEqual(parse_error(json.dumps({"error": "flat"})).message, "flat")

    def test_metadata_header_needs_all_three_keys(self):
        self.assertIsNone(metadata_header(None))
        self.assertIsNone(metadata_header({"user": "x"}))
        self.assertEqual(json.loads(metadata_header({"custom_param_a": "chat1", "other": 1})),
                         {"custom_param_a": "chat1", "custom_param_b": "na", "custom_param_c": "na"})


if __name__ == "__main__":
    unittest.main()
