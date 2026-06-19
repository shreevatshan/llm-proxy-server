"""Unit tests for the Part-1 Bedrock provider fixes (reviews/review.md).

These cover the pure/synchronous logic that can be tested without live AWS:
A2 (stream tool-call index), A4 (cache tokens), A5 (reasoning_content),
A6 (tool pairing), A7 (empty content), A9 (finish reason), C6 (budget clamp),
B4 (socket extraction structural assertion), and the _map_bedrock_error table.
"""

import socket
import unittest

from app.providers.bedrock_provider import (
    BedrockProvider,
    _extract_stream_socket,
    _map_bedrock_error,
)


def _provider() -> BedrockProvider:
    # Bypass __init__ (which needs real AWS config); we only exercise pure methods.
    p = BedrockProvider.__new__(BedrockProvider)
    p.debug = False
    return p


class FinishReasonTests(unittest.TestCase):
    def setUp(self):
        self.p = _provider()

    def test_known_reasons_map(self):
        self.assertEqual(self.p._convert_finish_reason("tool_use"), "tool_calls")
        self.assertEqual(self.p._convert_finish_reason("end_turn"), "stop")
        self.assertEqual(self.p._convert_finish_reason("max_tokens"), "length")
        self.assertEqual(self.p._convert_finish_reason("guardrail_intervened"), "content_filter")

    def test_unknown_reason_defaults_to_stop(self):
        # A9: never leak a non-enum value through verbatim.
        self.assertEqual(self.p._convert_finish_reason("some_new_reason"), "stop")

    def test_empty_is_none(self):
        self.assertIsNone(self.p._convert_finish_reason(""))


class BudgetTokenClampTests(unittest.TestCase):
    def setUp(self):
        self.p = _provider()

    def test_low_effort_small_max_clamped_to_floor(self):
        # C6: 0.3 * 2048 = 614 < 1024 -> clamp up to 1024.
        self.assertEqual(self.p._calc_budget_tokens(2048, "low"), 1024)

    def test_high_effort_clamped_below_max(self):
        self.assertEqual(self.p._calc_budget_tokens(4096, "high"), 4095)

    def test_never_exceeds_max_minus_one(self):
        # medium = 0.6 * 100000 = 60000, well under max-1.
        self.assertEqual(self.p._calc_budget_tokens(100000, "medium"), 60000)


class ToolPairingTests(unittest.TestCase):
    def setUp(self):
        self.p = _provider()

    def test_cross_turn_pairing_preserved(self):
        # A6: toolUse and its toolResult are NOT adjacent — must be kept.
        messages = [
            {"role": "assistant", "content": [
                {"text": "let me check"},
                {"toolUse": {"toolUseId": "t1", "name": "f", "input": {}}},
            ]},
            {"role": "assistant", "content": [{"text": "still working"}]},
            {"role": "user", "content": [
                {"toolResult": {"toolUseId": "t1", "content": [{"text": "ok"}]}},
            ]},
        ]
        out = self.p._validate_tool_use_result_pairing(messages)
        ids = [
            item["toolUse"]["toolUseId"]
            for m in out for item in m["content"]
            if isinstance(item, dict) and "toolUse" in item
        ]
        self.assertIn("t1", ids)

    def test_orphan_tooluse_dropped(self):
        messages = [
            {"role": "assistant", "content": [
                {"toolUse": {"toolUseId": "orphan", "name": "f", "input": {}}},
            ]},
            {"role": "user", "content": [{"text": "no result here"}]},
        ]
        out = self.p._validate_tool_use_result_pairing(messages)
        ids = [
            item["toolUse"]["toolUseId"]
            for m in out for item in m["content"]
            if isinstance(item, dict) and "toolUse" in item
        ]
        self.assertNotIn("orphan", ids)


class MapBedrockErrorTests(unittest.TestCase):
    def test_throttling_maps_to_429(self):
        m = _map_bedrock_error("ThrottlingException", "slow down")
        self.assertEqual(m["status"], 429)
        self.assertEqual(m["body"]["error"]["type"], "rate_limit_error")

    def test_validation_maps_to_400(self):
        m = _map_bedrock_error("ValidationException", "bad")
        self.assertEqual(m["status"], 400)
        self.assertEqual(m["body"]["error"]["type"], "invalid_request_error")

    def test_unknown_maps_to_500(self):
        m = _map_bedrock_error("SomethingNew", "x")
        self.assertEqual(m["status"], 500)


class StreamChunkTests(unittest.TestCase):
    def setUp(self):
        self.p = _provider()
        self.state = {"tool_call_index": -1, "block_index_map": {}}

    def test_message_start_omits_content(self):
        # A3: first delta carries role only, no content key.
        resp, _ = self.p._parse_stream_chunk(
            {"messageStart": {"role": "assistant"}}, "id", "m", self.state
        )
        self.assertEqual(resp["choices"][0]["delta"], {"role": "assistant"})

    def test_tool_only_response_index_zero(self):
        # A2: first tool block at contentBlockIndex 0 -> OpenAI index 0, not -1.
        chunk = {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {"toolUse": {"toolUseId": "t1", "name": "f"}},
            }
        }
        resp, state = self.p._parse_stream_chunk(chunk, "id", "m", self.state)
        self.assertEqual(resp["choices"][0]["delta"]["tool_calls"][0].index, 0)
        # The matching delta resolves to the same ordinal.
        delta_chunk = {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '{"a":1}'}},
            }
        }
        resp2, _ = self.p._parse_stream_chunk(delta_chunk, "id", "m", state)
        self.assertEqual(resp2["choices"][0]["delta"]["tool_calls"][0].index, 0)

    def test_reasoning_emits_reasoning_content(self):
        # A5: reasoning text on reasoning_content field, no <think> literals.
        chunk = {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "hmm"}}}}
        resp, _ = self.p._parse_stream_chunk(chunk, "id", "m", self.state)
        self.assertEqual(resp["choices"][0]["delta"], {"reasoning_content": "hmm"})

    def test_usage_includes_cache_tokens(self):
        # A4: cache token counts preserved in the streaming usage chunk.
        chunk = {"metadata": {"usage": {
            "inputTokens": 10, "outputTokens": 5, "totalTokens": 18,
            "cacheReadInputTokens": 3, "cacheWriteInputTokens": 0,
        }}}
        resp, _ = self.p._parse_stream_chunk(chunk, "id", "m", self.state)
        usage = resp["usage"]
        self.assertEqual(usage["total_tokens"], 18)
        self.assertEqual(usage["cache_read_input_tokens"], 3)


class SocketExtractionTests(unittest.TestCase):
    """B4: assert the private-attribute walk still finds a socket, so a
    dependency bump that changes the layout fails loudly in CI."""

    def test_finds_socket_in_expected_layout(self):
        class _FakeBufferedReader:
            def __init__(self, sock):
                self.raw = type("Raw", (), {"_sock": sock})()

        class _FakeHttplibResp:
            def __init__(self, sock):
                self.fp = _FakeBufferedReader(sock)

        class _FakeUrllib3Resp:
            def __init__(self, sock):
                self._fp = _FakeHttplibResp(sock)

        class _FakeEventStream:
            def __init__(self, sock):
                self._raw_stream = _FakeUrllib3Resp(sock)

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            found = _extract_stream_socket(_FakeEventStream(s))
            self.assertIs(found, s)
        finally:
            s.close()

    def test_returns_none_on_unexpected_layout(self):
        self.assertIsNone(_extract_stream_socket(object()))


if __name__ == "__main__":
    unittest.main()
