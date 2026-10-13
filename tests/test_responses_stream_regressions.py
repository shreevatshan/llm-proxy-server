"""Responses → Anthropic streaming: interleaved function calls and terminals.

Only the first call streams live; later calls and content that arrives while
it is open are buffered and emitted afterwards, so every delta lands on the
block it belongs to and each call's arguments are reconstructed exactly once.
"""

import unittest

from app.conversion.anthropic_openai import ResponsesToAnthropicConverter, StreamConversionState
from app.providers import anthropic_adapter
from app.providers.anthropic_adapter import anthropic_adapter_messages_stream
from tests import test_anthropic_compat as compat


def _added(index, call_id, name, item_id=None):
    item = {"type": "function_call", "call_id": call_id, "name": name}
    if item_id:
        item["id"] = item_id
    return ("response.output_item.added", {"output_index": index, "item": item})


def _args(index, delta, **extra):
    return ("response.function_call_arguments.delta", {"output_index": index, "delta": delta, **extra})


def _args_done(index, arguments):
    return ("response.function_call_arguments.done", {"output_index": index, "arguments": arguments})


def _item_done(index, call_id, name, arguments):
    return ("response.output_item.done", {"output_index": index, "item": {
        "type": "function_call", "call_id": call_id, "name": name, "arguments": arguments,
    }})


COMPLETED = ("response.completed", {"response": {"status": "completed"}})


class ResponsesStreamInterleavingTests(unittest.TestCase):
    def _stream(self, events):
        conv = ResponsesToAnthropicConverter()
        state = StreamConversionState(message_id="msg_1", model="m")
        out = []
        for event_type, data in events:
            out.extend(conv.convert_stream_event(event_type, data, state))
        out.extend(conv.convert_stream_end(state))
        events = compat._parse_sse(out)
        compat.StreamConversionTests._assert_well_formed(self, events)
        return events

    def _tool_inputs(self, events):
        return compat.StreamConversionTests._tool_inputs(self, events)

    @staticmethod
    def _starts(events):
        return [d["content_block"]["type"] for t, d in events if t == "content_block_start"]

    @staticmethod
    def _stop_reason(events):
        return [d for t, d in events if t == "message_delta"][0]["delta"]["stop_reason"]

    def _assert_error_only(self, events, message=None):
        types = [t for t, _ in events]
        self.assertEqual(types[-1], "error")
        self.assertNotIn("message_delta", types)
        self.assertNotIn("message_stop", types)
        if message is not None:
            self.assertIn(message, events[-1][1]["error"]["message"])

    def test_alternating_call_fragments(self):
        events = self._stream([
            _added(0, "call_a", "f"),
            _added(1, "call_b", "g"),
            _args(0, '{"x"'),
            _args(1, '{"y"'),
            _args(0, ": 1}"),
            _args(1, ": 2}"),
            _args_done(0, '{"x": 1}'),
            _args_done(1, '{"y": 2}'),
            COMPLETED,
        ])
        self.assertEqual(self._tool_inputs(events), [("f", {"x": 1}), ("g", {"y": 2})])
        self.assertEqual(self._stop_reason(events), "tool_use")

    def test_later_call_announced_before_first_fragment(self):
        events = self._stream([
            _added(0, "call_a", "f"),
            _added(1, "call_b", "g"),
            _args(1, '{"y": 2}'),
            _args(0, '{"x": 1}'),
            _item_done(1, "call_b", "g", '{"y": 2}'),
            _item_done(0, "call_a", "f", '{"x": 1}'),
            COMPLETED,
        ])
        self.assertEqual(self._tool_inputs(events), [("f", {"x": 1}), ("g", {"y": 2})])

    def test_arguments_before_metadata_are_buffered_by_alias(self):
        events = self._stream([
            _args(0, '{"x":', item_id="fc_a"),
            ("response.output_item.added", {"output_index": 0, "item": {
                "type": "function_call", "id": "fc_a", "call_id": "call_a", "name": "f",
            }}),
            ("response.function_call_arguments.delta", {"item_id": "fc_a", "delta": " 1}"}),
            COMPLETED,
        ])
        self.assertEqual(self._tool_inputs(events), [("f", {"x": 1})])
        start = [d for t, d in events if t == "content_block_start"][0]["content_block"]
        self.assertEqual(start["id"], "call_a")

    def test_unknown_index_is_not_assigned_to_live_block(self):
        events = self._stream([
            _added(0, "call_a", "f"),
            _args(7, '{"stray": true}'),
            _args(0, "{}"),
            COMPLETED,
        ])
        self.assertEqual(self._tool_inputs(events), [("f", {})])

    def test_content_between_tool_fragments_is_deferred(self):
        events = self._stream([
            _added(0, "call_a", "f"),
            _args(0, '{"a"'),
            ("response.output_text.delta", {"delta": "hi"}),
            ("response.reasoning_summary_text.delta", {"delta": "why"}),
            ("response.refusal.delta", {"delta": "!"}),
            _args(0, ": 1}"),
            _args_done(0, '{"a": 1}'),
            COMPLETED,
        ])
        self.assertEqual(self._tool_inputs(events), [("f", {"a": 1})])
        self.assertEqual(self._starts(events), ["tool_use", "text", "thinking", "text"])
        text = "".join(d["delta"].get("text", "") for t, d in events if t == "content_block_delta")
        self.assertEqual(text, "hi!")

    def test_deferred_summary_parts_keep_separator(self):
        events = self._stream([
            _added(0, "call_a", "f"),
            ("response.reasoning_summary_part.added", {"summary_index": 0}),
            ("response.reasoning_summary_text.delta", {"delta": "a"}),
            ("response.reasoning_summary_part.added", {"summary_index": 1}),
            ("response.reasoning_summary_text.delta", {"delta": "b"}),
            _args_done(0, "{}"),
            COMPLETED,
        ])
        self.assertEqual(self._starts(events), ["tool_use", "thinking"])
        thinking = "".join(d["delta"].get("thinking", "") for t, d in events if t == "content_block_delta")
        self.assertEqual(thinking, "a\n\nb")

    def test_out_of_order_done_only_calls(self):
        events = self._stream([
            _added(0, "call_a", "f"),
            _added(1, "call_b", "g"),
            _item_done(1, "call_b", "g", '{"b": 2}'),
            _args_done(0, '{"a": 1}'),
            _item_done(0, "call_a", "f", '{"a": 1}'),
            COMPLETED,
        ])
        self.assertEqual(self._tool_inputs(events), [("f", {"a": 1}), ("g", {"b": 2})])

    def test_done_snapshot_completes_prefix_without_duplication(self):
        events = self._stream([
            _added(0, "call_a", "f"),
            _args(0, '{"a":'),
            _args_done(0, '{"a": 1}'),
            _item_done(0, "call_a", "f", '{"a": 1}'),
            ("response.completed", {"response": {"status": "completed", "output": [
                {"type": "function_call", "call_id": "call_a", "name": "f", "arguments": '{"a": 1}'},
            ]}}),
        ])
        self.assertEqual(self._tool_inputs(events), [("f", {"a": 1})])

    def test_differently_formatted_equal_snapshot_is_accepted(self):
        events = self._stream([
            _added(0, "call_a", "f"),
            _args(0, '{"a":1,"b":[2]}'),
            _args_done(0, '{"a": 1, "b": [2]}'),
            COMPLETED,
        ])
        self.assertEqual(self._tool_inputs(events), [("f", {"a": 1, "b": [2]})])
        self.assertEqual(self._stop_reason(events), "tool_use")

    def test_conflicting_snapshot_after_output_fails_typed(self):
        events = self._stream([
            _added(0, "call_a", "f"),
            _args(0, '{"a": 1}'),
            _args_done(0, '{"b": 2}'),
            COMPLETED,
        ])
        self._assert_error_only(events, "conflicting arguments")
        self.assertEqual(events[-1][1]["error"]["type"], "api_error")

    def test_failure_is_sticky_and_buffered_calls_are_not_flushed(self):
        events = self._stream([
            _added(0, "call_a", "f"),
            _added(1, "call_b", "g"),
            _args(1, '{"y": 2}'),
            ("response.failed", {"response": {"status": "failed", "error": {"code": "server_error", "message": "boom"}}}),
            ("response.completed", {"response": {"status": "completed"}}),
        ])
        self._assert_error_only(events, "boom")
        self.assertEqual(self._starts(events), ["tool_use"])

    def test_error_event_then_completion_still_errors(self):
        events = self._stream([
            ("response.output_text.delta", {"delta": "par"}),
            ("error", {"error": {"code": "rate_limit_exceeded", "message": "slow"}}),
            COMPLETED,
        ])
        self._assert_error_only(events, "slow")
        self.assertEqual(events[-1][1]["error"]["type"], "rate_limit_error")

    def test_incomplete_keeps_stop_reason_and_partial_calls(self):
        events = self._stream([
            _added(0, "call_a", "f"),
            _args(0, '{"a": 1}'),
            _added(1, "call_b", "g"),
            _args(1, '{"b": 2}'),
            ("response.incomplete", {"response": {"status": "incomplete",
                                                  "incomplete_details": {"reason": "max_output_tokens"},
                                                  "usage": {"input_tokens": 3, "output_tokens": 5}}}),
        ])
        self.assertEqual(self._tool_inputs(events), [("f", {"a": 1}), ("g", {"b": 2})])
        delta = [d for t, d in events if t == "message_delta"][0]
        self.assertEqual(delta["delta"]["stop_reason"], "max_tokens")
        self.assertEqual(delta["usage"], {"input_tokens": 3, "output_tokens": 5})

    def test_sequential_controls_unchanged(self):
        events = self._stream([
            ("response.reasoning_summary_text.delta", {"delta": "think"}),
            ("response.output_text.delta", {"delta": "answer"}),
            _added(1, "call_a", "f"),
            _args(1, '{"q": "x"}'),
            _item_done(1, "call_a", "f", '{"q": "x"}'),
            COMPLETED,
        ])
        self.assertEqual(self._starts(events), ["thinking", "text", "tool_use"])
        self.assertEqual(self._tool_inputs(events), [("f", {"q": "x"})])


class ResponsesAdapterStreamTests(unittest.TestCase):
    def setUp(self):
        anthropic_adapter._rejected_params.clear()
        self.addCleanup(anthropic_adapter._rejected_params.clear)

    def test_adapter_interleaved_calls_and_text(self):
        provider = compat._custom_provider()
        sse = compat._sse_event

        async def responses_create_stream(request):
            yield sse("response.created", {"response": {"status": "in_progress"}})
            yield sse("response.output_item.added", {"output_index": 0, "item": {"type": "function_call", "call_id": "call_a", "name": "f"}})
            yield sse("response.output_item.added", {"output_index": 1, "item": {"type": "function_call", "call_id": "call_b", "name": "g"}})
            yield sse("response.function_call_arguments.delta", {"output_index": 1, "delta": '{"y":'})
            yield sse("response.function_call_arguments.delta", {"output_index": 0, "delta": '{"x":'})
            yield sse("response.output_text.delta", {"delta": "note"})
            yield sse("response.function_call_arguments.delta", {"output_index": 1, "delta": " 2}"})
            yield sse("response.function_call_arguments.delta", {"output_index": 0, "delta": " 1}"})
            yield sse("response.function_call_arguments.done", {"output_index": 0, "arguments": '{"x": 1}'})
            yield sse("response.function_call_arguments.done", {"output_index": 1, "arguments": '{"y": 2}'})
            yield sse("response.completed", {"response": {"status": "completed", "usage": {"input_tokens": 4, "output_tokens": 6}}})

        provider.responses_create_stream = responses_create_stream
        chunks = compat.run(compat._collect(anthropic_adapter_messages_stream(
            provider, compat._req(stream=True), transport="responses",
        )))
        events = compat._parse_sse(chunks)
        compat.StreamConversionTests._assert_well_formed(self, events)
        self.assertEqual(compat.StreamConversionTests._tool_inputs(self, events), [("f", {"x": 1}), ("g", {"y": 2})])
        types = [t for t, _ in events]
        self.assertEqual(types.count("message_start"), 1)
        self.assertEqual(types[-2:], ["message_delta", "message_stop"])
        self.assertEqual(events[-2][1]["delta"]["stop_reason"], "tool_use")


if __name__ == "__main__":
    unittest.main()
