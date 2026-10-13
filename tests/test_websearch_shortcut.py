"""The Claude Code WebSearch shortcut must honor the client's tool_choice.

A matching standalone search prompt is answered by searching before the
model runs, which bypasses the model's own tool selection; that is only
allowed when the request's tool_choice permits the search tool.
"""

import json
import unittest

from app.anthropic_models import AnthropicMessagesRequest
from app.websearch import maybe_intercept_anthropic, tools
from app.websearch.loop import iter_sse_events
from tests import test_adapter_regressions as adapter_tests
from tests import test_anthropic_compat as compat
from tests import test_websearch as ws

PROMPT = "Perform a web search for the query: weather in Paris"
NONE = {"type": "none"}


def _request(stream=False, **extra):
    body = {
        "model": "custom:test/claude",
        "max_tokens": 100,
        "stream": stream,
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
    }
    body.update(extra)
    return AnthropicMessagesRequest.model_validate(body)


class ShortcutPolicyTests(unittest.TestCase):
    def test_allowed_choices(self):
        for choice in (
            None,
            {"type": "auto"},
            {"type": "any"},
            {"type": "auto", "disable_parallel_tool_use": True},
            {"type": "tool", "name": "web_search"},
            {"type": "tool", "name": tools.INTERNAL_TOOL_NAME},
            {"type": "tool", "name": "litellm_web_search"},
        ):
            with self.subTest(choice=choice):
                self.assertTrue(tools.anthropic_search_shortcut_allowed(choice))
                # The validated model form behaves like the plain dict.
                validated = _request(tool_choice=choice).tool_choice if choice else None
                self.assertTrue(tools.anthropic_search_shortcut_allowed(validated))

    def test_disallowed_choices(self):
        for choice in (
            NONE,
            {"type": "tool", "name": "get_weather"},
            {"type": "something_new"},
            "none",
            "auto",
        ):
            with self.subTest(choice=choice):
                self.assertFalse(tools.anthropic_search_shortcut_allowed(choice))
        validated = _request(tool_choice=NONE).tool_choice
        self.assertFalse(tools.anthropic_search_shortcut_allowed(validated))


class ShortcutLoopTests(ws._ConfigMixin, unittest.TestCase):
    def test_shortcut_runs_when_choice_allows_it(self):
        provider = ws._Provider(responses=[ws._msg([{"type": "text", "text": "Sunny."}], "end_turn")])
        loop = maybe_intercept_anthropic(_request(), provider)
        self.assertEqual(loop.shortcircuit_query, "weather in Paris")
        result = ws.run(loop.run())
        self.assertEqual(self._searches, [["weather in Paris"]])
        self.assertIn("web_search_tool_result", [b["type"] for b in result["content"]])

    def test_explicit_none_disables_the_shortcut(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self._searches.clear()
                text = [{"type": "text", "text": "No search."}]
                provider = ws._Provider(
                    responses=[ws._msg(text, "end_turn")],
                    streams=[ws._stream_round(text, "end_turn")],
                )
                loop = maybe_intercept_anthropic(_request(stream=stream, tool_choice=NONE), provider)
                self.assertIsNone(loop.shortcircuit_query)
                if stream:
                    body = "".join(ws.run(ws._collect(loop.stream())))
                    self.assertNotIn("web_search_tool_result", body)
                    usage = [p for e, p, _ in iter_sse_events(body) if e == "message_delta"][-1]["usage"]
                else:
                    result = ws.run(loop.run())
                    self.assertEqual([b["type"] for b in result["content"]], ["text"])
                    usage = result["usage"]
                self.assertNotIn("server_tool_use", usage)
                self.assertEqual(self._searches, [])
                # The model runs normally, with the client's choice and history.
                (sent,) = provider.requests
                self.assertEqual(tools._as_dict(sent.tool_choice), NONE)
                self.assertEqual(len(sent.messages), 1)
                self.assertEqual([t["name"] for t in sent.tools], [tools.INTERNAL_TOOL_NAME])

    def test_forced_unrelated_tool_disables_the_shortcut(self):
        weather = {"name": "get_weather", "input_schema": {"type": "object", "properties": {}}}
        request = _request(
            tools=[{"type": "web_search_20250305", "name": "web_search"}, weather],
            tool_choice={"type": "tool", "name": "get_weather"},
        )
        loop = maybe_intercept_anthropic(request, ws._Provider())
        self.assertIsNone(loop.shortcircuit_query)

    def test_dispatch_rechecks_the_choice(self):
        provider = ws._Provider(responses=[ws._msg([{"type": "text", "text": "No search."}], "end_turn")])
        loop = maybe_intercept_anthropic(_request(), provider)
        self.assertIsNotNone(loop.shortcircuit_query)
        loop.request = loop.request.model_copy(update={"tool_choice": NONE})
        ws.run(loop.run())
        self.assertEqual(self._searches, [])
        (sent,) = provider.requests
        self.assertEqual(len(sent.messages), 1)
        self.assertEqual(tools._as_dict(sent.tool_choice), NONE)


class ShortcutAdapterProviderTests(ws._ConfigMixin, unittest.TestCase):
    """Through a real adapter-mode custom provider: tool_choice none reaches
    the OpenAI-compatible upstream as "none" and no search runs."""

    def setUp(self):
        super().setUp()
        adapter_tests._AdapterTestCase.setUp(self)

    def test_none_reaches_the_upstream(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self._searches.clear()
                provider = compat._custom_provider(name="test")
                provider.client = adapter_tests.FakeClient()
                loop = maybe_intercept_anthropic(_request(stream=stream, tool_choice=NONE), provider)
                if stream:
                    ws.run(ws._collect(loop.stream()))
                else:
                    ws.run(loop.run())
                self.assertEqual(self._searches, [])
                (kwargs,) = provider.client.kwargs()
                self.assertEqual(kwargs["tool_choice"], "none")
                self.assertEqual(len(kwargs["messages"]), 1)
                self.assertEqual(json.dumps(kwargs["messages"][0]["content"]).count(PROMPT), 1)


if __name__ == "__main__":
    unittest.main()
