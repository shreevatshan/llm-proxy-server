"""Tests for web search interception (SearXNG and 4get backends)."""

import asyncio
import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import httpx

from app.anthropic_models import AnthropicMessagesRequest
from app.providers.base import ProviderHTTPError
from app.openai_models import ChatCompletionRequest, ChatCompletionResponse, ResponseObject, ResponsesCreateRequest
from app.websearch import (
    maybe_intercept_anthropic,
    maybe_intercept_chat,
    maybe_intercept_responses,
    responses_websearch_stream,
    rewrite_anthropic_count_tokens_payload,
    settle_pending_responses_calls,
    synthesize_responses_stream,
)
from app.websearch import backends, client, fourget, results, searxng, tools
from app.websearch.loop import LIMIT_NUDGE, LIMIT_REACHED_TEXT, add_usage, iter_sse_events
from app.websearch.results import SearchFailed, SearchResult, SearchSucceeded
from app.websearch.settings import WebSearchConfig, websearch_settings_cache


def run(coro):
    return asyncio.run(coro)


CFG = WebSearchConfig(
    enabled=True,
    searxng_base_url="http://searx.test",
    max_results=2,
    max_snippet_chars=40,
    max_agentic_loops=2,
    enabled_providers=frozenset({"custom:test"}),
)

SEARX_PAYLOAD = {
    "results": [
        {"title": "One", "url": "https://a.example/1", "content": "first snippet " * 10, "publishedDate": "2026-01-01"},
        {"title": "Dup", "url": "https://a.example/1", "content": "dup"},
        {"title": "Two", "url": "https://b.example/2", "content": "second"},
        {"title": "Three", "url": "https://c.example/3", "content": "third"},
    ]
}


def fake_search_outcome(query, results=None):
    return SearchSucceeded(query, results or [SearchResult("T", f"https://x.example/{query}", f"about {query}")])


class _ConfigMixin:
    def setUp(self):
        self._saved = websearch_settings_cache.config
        websearch_settings_cache.set_config(CFG)
        self._searches = []

        async def fake_run_searches(queries, cfg):
            self._searches.append(list(queries))
            return [fake_search_outcome(q) for q in queries]

        patcher = patch("app.websearch.loop.run_searches", fake_run_searches)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        websearch_settings_cache.set_config(self._saved)


# ==================== SearXNG client ====================

class SearxngTests(unittest.TestCase):
    def _with_transport(self, handler):
        transport = httpx.MockTransport(handler)
        client._client = httpx.AsyncClient(transport=transport)
        self.addCleanup(lambda: run(client.close_client()))

    def test_parse_dedupe_truncate(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            return httpx.Response(200, json=SEARX_PAYLOAD)

        self._with_transport(handler)
        cfg = CFG.with_overrides(engines="bing", time_range="day")
        outcome = run(searxng.search("hello world", cfg))
        self.assertTrue(outcome.ok)
        self.assertEqual([r.url for r in outcome.results], ["https://a.example/1", "https://b.example/2"])
        self.assertLessEqual(len(outcome.results[0].snippet), 40)
        self.assertEqual(outcome.results[0].date, "2026-01-01")
        self.assertIn("/search?", seen["url"])
        self.assertIn("format=json", seen["url"])
        self.assertIn("engines=bing", seen["url"])
        self.assertIn("time_range=day", seen["url"])
        self.assertIsNone(seen.get("auth"))

    def test_engines_and_categories_are_exclusive(self):
        seen = []
        self._with_transport(lambda request: seen.append(str(request.url)) or httpx.Response(200, json=SEARX_PAYLOAD))
        run(searxng.search("q", CFG.with_overrides(engines="bing", categories="general")))
        run(searxng.search("q", CFG.with_overrides(engines=None, categories="news,it")))
        self.assertIn("engines=bing", seen[0])
        self.assertNotIn("categories=", seen[0])
        self.assertIn("categories=news%2Cit", seen[1])
        self.assertNotIn("engines=", seen[1])

    def test_error_mapping(self):
        for status, code in ((429, "too_many_requests"), (403, "unavailable"), (500, "unavailable")):
            self._with_transport(lambda request, s=status: httpx.Response(s, text="nope"))
            outcome = run(searxng.search("q", CFG))
            self.assertFalse(outcome.ok)
            self.assertEqual(outcome.error_code, code)
        self._with_transport(lambda request: httpx.Response(200, text="<html>"))
        outcome = run(searxng.search("q", CFG))
        self.assertIn("json", outcome.message)

    def test_timeout(self):
        def handler(request):
            raise httpx.ReadTimeout("slow")

        self._with_transport(handler)
        self.assertEqual(run(searxng.search("q", CFG)).error_code, "unavailable")

    def test_fetch_engines(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            return httpx.Response(200, json={
                "categories": ["general", "it"],
                "engines": [
                    {"name": "duckduckgo", "categories": ["general", "web"], "enabled": True, "shortcut": "ddg"},
                    {"name": "bing", "categories": ["general"], "enabled": True, "shortcut": "bi"},
                    {"name": "github", "categories": ["it"], "enabled": False},
                    "junk",
                ],
            })

        self._with_transport(handler)
        data = run(searxng.fetch_engines("http://h:8080/search", 5))
        self.assertEqual(seen["url"], "http://h:8080/config")
        self.assertEqual([e["name"] for e in data["engines"]], ["bing", "duckduckgo", "github"])
        self.assertFalse(data["engines"][2]["enabled"])
        self.assertEqual(data["categories"], ["general", "it"])

    def test_fetch_engines_errors(self):
        self._with_transport(lambda request: httpx.Response(404))
        with self.assertRaises(searxng.SearxngConfigError):
            run(searxng.fetch_engines("http://h", 5))
        self._with_transport(lambda request: httpx.Response(200, text="<html>"))
        with self.assertRaises(searxng.SearxngConfigError):
            run(searxng.fetch_engines("http://h", 5))

    def test_search_url(self):
        self.assertEqual(searxng.search_url("http://h:8080/"), "http://h:8080/search")
        self.assertEqual(searxng.search_url("http://h/search"), "http://h/search")

    def test_format_text(self):
        text = results.format_outcome_text(SearchSucceeded("q", [SearchResult("T", "https://u", "s", "d")]))
        self.assertEqual(text, "Title: T\nURL: https://u\nDate: d\nSnippet: s")
        self.assertTrue(results.format_outcome_text(SearchFailed("q", "unavailable", "down")).startswith("Search failed"))


# ==================== Tool detection / rewriting ====================

# ==================== 4get client ====================

FOURGET_CFG = CFG.with_overrides(
    provider="fourget",
    searxng_base_url=None,
    fourget_base_url="http://fourget.test",
    fourget_scraper="ddg",
)

FOURGET_PAYLOAD = {
    "status": "ok",
    "spelling": {"type": "no_correction"},
    "web": [
        {"title": "One", "description": "first snippet " * 10, "url": "https://a.example/1", "date": 1767225600},
        {"title": "Dup", "description": "dup", "url": "https://a.example/1", "date": None},
        {"title": "Two", "description": "second", "url": "https://b.example/2", "date": None},
        {"title": "Three", "description": "third", "url": "https://c.example/3", "date": None},
    ],
    "image": [], "video": [], "news": [], "related": [], "answer": [],
}


class FourgetTests(unittest.TestCase):
    def _with_transport(self, handler):
        transport = httpx.MockTransport(handler)
        client._client = httpx.AsyncClient(transport=transport)
        self.addCleanup(lambda: run(client.close_client()))

    def test_api_url(self):
        self.assertEqual(fourget.api_url("http://h/"), "http://h/api/v1/web")
        self.assertEqual(fourget.api_url("http://h/api/v1/web"), "http://h/api/v1/web")

    def test_parse_dedupe_truncate(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            return httpx.Response(200, json=FOURGET_PAYLOAD)

        self._with_transport(handler)
        outcome = run(fourget.search("hello world", FOURGET_CFG.with_overrides(time_range="week")))
        self.assertTrue(outcome.ok)
        self.assertEqual([r.url for r in outcome.results], ["https://a.example/1", "https://b.example/2"])
        self.assertLessEqual(len(outcome.results[0].snippet), 40)
        self.assertEqual(outcome.results[0].date, "2026-01-01")
        self.assertIn("/api/v1/web?", seen["url"])
        self.assertIn("s=hello+world", seen["url"])
        self.assertIn("scraper=ddg", seen["url"])
        self.assertIn("nsfw=maybe", seen["url"])  # safesearch 1
        expected = (datetime.now(timezone.utc).date() - timedelta(days=7)).isoformat()
        self.assertIn(f"newer={expected}", seen["url"])

    def test_optional_params_are_omitted(self):
        seen = []
        self._with_transport(lambda request: seen.append(str(request.url)) or httpx.Response(200, json=FOURGET_PAYLOAD))
        run(fourget.search("q", FOURGET_CFG.with_overrides(fourget_scraper=None, safesearch=0)))
        self.assertNotIn("scraper=", seen[0])
        self.assertNotIn("newer=", seen[0])
        self.assertNotIn("lang=", seen[0])
        self.assertNotIn("country=", seen[0])
        self.assertIn("nsfw=yes", seen[0])

    def test_error_status_on_http_200(self):
        # 4get never sets a status code; failures arrive as 200 + a status field.
        self._with_transport(lambda request: httpx.Response(200, json={"status": "Invalid scraper"}))
        outcome = run(fourget.search("q", FOURGET_CFG))
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.error_code, "unavailable")
        self.assertIn("Invalid scraper", outcome.message)

    def test_error_mapping(self):
        self._with_transport(lambda request: httpx.Response(429, json={"status": "slow down"}))
        self.assertEqual(run(fourget.search("q", FOURGET_CFG)).error_code, "too_many_requests")
        self._with_transport(lambda request: httpx.Response(502, text="bad gateway"))
        self.assertEqual(run(fourget.search("q", FOURGET_CFG)).error_code, "unavailable")
        self._with_transport(lambda request: httpx.Response(200, text="not json"))
        self.assertEqual(run(fourget.search("q", FOURGET_CFG)).error_code, "unavailable")

    def test_non_string_fields_fall_back(self):
        payload = {"status": "ok", "web": [
            {"url": "https://a.example/1", "title": 42, "description": ["x"]},
        ]}
        self._with_transport(lambda request: httpx.Response(200, json=payload))
        outcome = run(fourget.search("q", FOURGET_CFG))
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.results[0].title, "https://a.example/1")
        self.assertEqual(outcome.results[0].snippet, "")

    def test_malformed_payload_fails_without_raising(self):
        self._with_transport(lambda request: httpx.Response(200, json={"status": "ok", "web": 5}))
        outcome = run(fourget.search("q", FOURGET_CFG))
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.error_code, "unavailable")

    def test_timeout_and_missing_url(self):
        def handler(request):
            raise httpx.TimeoutException("timed out")

        self._with_transport(handler)
        self.assertEqual(run(fourget.search("q", FOURGET_CFG)).error_code, "unavailable")
        self.assertEqual(
            run(fourget.search("q", FOURGET_CFG.with_overrides(fourget_base_url=None))).error_code,
            "unavailable",
        )

    def test_date_variants(self):
        # Usually a unix timestamp, sometimes an upstream string, often null.
        self.assertEqual(fourget._format_date(1767225600), "2026-01-01")
        self.assertEqual(fourget._format_date("1767225600"), "2026-01-01")
        self.assertEqual(fourget._format_date("1767225600.0"), "2026-01-01")
        self.assertEqual(fourget._format_date(1767225600.5), "2026-01-01")
        self.assertEqual(fourget._format_date("last tuesday"), "last tuesday")
        self.assertIsNone(fourget._format_date(None))
        self.assertIsNone(fourget._format_date(""))
        self.assertIsNone(fourget._format_date(True))
        self.assertIsNone(fourget._format_date(10 ** 20))


class BackendDispatchTests(unittest.TestCase):
    def test_routes_by_provider(self):
        calls = []

        async def fake(name, query, cfg):
            calls.append(name)
            return SearchSucceeded(query)

        with patch("app.websearch.searxng.search", lambda q, c: fake("searxng", q, c)), \
             patch("app.websearch.fourget.search", lambda q, c: fake("fourget", q, c)):
            run(backends.search("q", CFG))
            run(backends.search("q", FOURGET_CFG))
            run(backends.run_searches(["a", "b"], FOURGET_CFG))
        self.assertEqual(calls, ["searxng", "fourget", "fourget", "fourget"])

    def test_rejects_bad_queries_before_dispatch(self):
        with patch("app.websearch.searxng.search") as sx, patch("app.websearch.fourget.search") as fg:
            for cfg in (CFG, FOURGET_CFG):
                self.assertEqual(run(backends.search("  ", cfg)).error_code, "invalid_tool_input")
                self.assertEqual(run(backends.search("x" * 600, cfg)).error_code, "query_too_long")
        sx.assert_not_called()
        fg.assert_not_called()


class ToolTests(unittest.TestCase):
    def test_anthropic_detection(self):
        self.assertTrue(tools.is_web_search_tool_anthropic({"type": "web_search_20250305", "name": "web_search"}))
        self.assertTrue(tools.is_web_search_tool_anthropic({"name": "litellm_web_search", "input_schema": {}}))
        # Client-executed tools are left alone.
        self.assertFalse(tools.is_web_search_tool_anthropic({"name": "WebSearch", "input_schema": {"type": "object"}}))
        self.assertFalse(tools.is_web_search_tool_anthropic({"name": "web_search", "input_schema": {"type": "object"}}))
        self.assertFalse(tools.is_web_search_tool_anthropic({"type": "web_fetch_20250910", "name": "web_fetch"}))

    def test_anthropic_rewrite_and_tool_choice(self):
        out, found, native = tools.rewrite_anthropic_tools([
            {"name": "get_weather", "input_schema": {"type": "object"}},
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
        ])
        self.assertTrue(found and native)
        self.assertEqual([t["name"] for t in out], ["get_weather", tools.INTERNAL_TOOL_NAME])
        self.assertEqual(
            tools.remap_anthropic_tool_choice({"type": "tool", "name": "web_search"}),
            {"type": "tool", "name": tools.INTERNAL_TOOL_NAME},
        )

    def test_chat_detection(self):
        self.assertTrue(tools.detect_chat({"web_search_options": {}}))
        self.assertTrue(tools.detect_chat({"tools": [{"type": "function", "function": {"name": "litellm_web_search"}}]}))
        # A client function plainly named web_search is the client's own tool.
        self.assertFalse(tools.detect_chat({"tools": [{"type": "function", "function": {"name": "web_search"}}]}))
        out = tools.rewrite_chat_request({"web_search_options": {}, "tools": None})
        self.assertNotIn("web_search_options", out)
        self.assertEqual(out["tools"][0]["function"]["name"], tools.INTERNAL_TOOL_NAME)

    def test_responses_rewrite(self):
        out = tools.rewrite_responses_request({
            "tools": [{"type": "web_search_preview"}, {"type": "function", "name": "f"}],
            "tool_choice": {"type": "web_search_preview"},
            "include": ["web_search_call.action.sources"],
            "input": [{"type": "web_search_call", "id": "ws_1"}, {"role": "user", "content": "hi"}],
        })
        self.assertEqual([t.get("name") for t in out["tools"]], ["f", tools.INTERNAL_TOOL_NAME])
        self.assertEqual(out["tool_choice"], {"type": "function", "name": tools.INTERNAL_TOOL_NAME})
        self.assertIsNone(out["include"])
        self.assertEqual(out["input"], [{"role": "user", "content": "hi"}])

    def test_native_block_shape_golden(self):
        outcome = SearchSucceeded("q", [SearchResult("Title", "https://u.example", "snip", "2026-01-02")])
        server_use, result = tools.build_native_blocks("srvtoolu_wspabc", outcome)
        self.assertEqual(server_use, {"type": "server_tool_use", "id": "srvtoolu_wspabc", "name": "web_search", "input": {"query": "q"}})
        self.assertEqual(result["type"], "web_search_tool_result")
        self.assertEqual(result["tool_use_id"], "srvtoolu_wspabc")
        item = result["content"][0]
        self.assertEqual(set(item), {"type", "url", "title", "encrypted_content", "page_age"})
        self.assertEqual(item["type"], "web_search_result")
        _, err = tools.build_native_blocks("srvtoolu_wspx", SearchFailed("q", "unavailable", "down"))
        self.assertEqual(err["content"], {"type": "web_search_tool_result_error", "error_code": "unavailable"})

    def test_rehydration_splits_rounds_and_keeps_thinking(self):
        outcome = SearchSucceeded("q", [SearchResult("T", "https://u", "s")])
        native = tools.build_native_blocks("srvtoolu_wsp1", outcome)
        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "plan", "signature": "sig1"},
                {"type": "text", "text": "Searching."},
                *native,
                {"type": "text", "text": "Answer."},
            ]},
            {"role": "user", "content": "follow-up"},
        ]
        out = tools.rehydrate_anthropic_messages(messages)
        self.assertEqual([m["role"] for m in out], ["user", "assistant", "user", "assistant", "user"])
        self.assertEqual(out[1]["content"][0], {"type": "thinking", "thinking": "plan", "signature": "sig1"})
        self.assertEqual(out[1]["content"][2]["type"], "tool_use")
        self.assertEqual(out[1]["content"][2]["name"], tools.INTERNAL_TOOL_NAME)
        self.assertEqual(out[2]["content"][0]["type"], "tool_result")
        self.assertEqual(out[2]["content"][0]["content"], results.format_outcome_text(outcome))
        self.assertEqual(out[3]["content"], [{"type": "text", "text": "Answer."}])

    def test_rehydration_merges_trailing_results_with_next_user(self):
        native = tools.build_native_blocks("srvtoolu_wsp2", SearchSucceeded("q", []))
        out = tools.rehydrate_anthropic_messages([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "text", "text": "x"}, *native]},
            {"role": "user", "content": "more"},
        ])
        self.assertEqual([m["role"] for m in out], ["user", "assistant", "user"])
        self.assertEqual(out[2]["content"][0]["type"], "tool_result")
        self.assertEqual(out[2]["content"][1], {"type": "text", "text": "more"})

    def test_genuine_upstream_blocks_untouched(self):
        messages = [{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "srvtoolu_01real", "name": "web_search", "input": {"query": "q"}},
        ]}]
        self.assertIs(tools.rehydrate_anthropic_messages(messages), messages)

    def test_usage_sum(self):
        total = {}
        add_usage(total, {"input_tokens": 3, "output_tokens": 1, "details": {"cached": 2}})
        add_usage(total, {"input_tokens": 4, "output_tokens": 2, "details": {"cached": 1}})
        self.assertEqual(total, {"input_tokens": 7, "output_tokens": 3, "details": {"cached": 3}})


# ==================== Anthropic loop ====================

class _Provider:
    full_provider_name = "custom:test"

    def __init__(self, responses=None, streams=None):
        self.responses = list(responses or [])
        self.streams = list(streams or [])
        self.requests = []

    async def anthropic_messages(self, request, anthropic_beta=None):
        self.requests.append(request)
        return self.responses.pop(0)

    async def anthropic_messages_stream(self, request, anthropic_beta=None):
        self.requests.append(request)
        for chunk in self.streams.pop(0):
            yield chunk


def _msg(content, stop_reason, usage=None):
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
        "content": content, "stop_reason": stop_reason,
        "usage": usage or {"input_tokens": 10, "output_tokens": 5},
    }


def _tool_use(query, tid="toolu_1"):
    return {"type": "tool_use", "id": tid, "name": tools.INTERNAL_TOOL_NAME, "input": {"query": query}}


def _anthropic_request(**extra):
    body = {
        "model": "custom:test/claude",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "news?"}],
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
    }
    body.update(extra)
    return AnthropicMessagesRequest.model_validate(body)


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_round(blocks, stop_reason, input_tokens=10, output_tokens=5):
    events = [sse("message_start", {"type": "message_start", "message": _msg([], None, {"input_tokens": input_tokens, "output_tokens": 1})})]
    for i, block in enumerate(blocks):
        if block["type"] == "text":
            events.append(sse("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}))
            events.append(sse("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": block["text"]}}))
        elif block["type"] == "thinking":
            events.append(sse("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "thinking", "thinking": ""}}))
            events.append(sse("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "thinking_delta", "thinking": block["thinking"]}}))
            events.append(sse("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "signature_delta", "signature": block["signature"]}}))
        else:
            events.append(sse("content_block_start", {"type": "content_block_start", "index": i, "content_block": {**block, "input": {}}}))
            events.append(sse("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}}))
        events.append(sse("content_block_stop", {"type": "content_block_stop", "index": i}))
    events.append(sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": output_tokens}}))
    events.append(sse("message_stop", {"type": "message_stop"}))
    return events


async def _collect(gen):
    return [chunk async for chunk in gen]


class AnthropicLoopTests(_ConfigMixin, unittest.TestCase):
    def test_not_intercepted_when_provider_not_enabled(self):
        provider = _Provider()
        provider.full_provider_name = "other:x"
        self.assertIsNone(maybe_intercept_anthropic(_anthropic_request(), provider))

    def test_not_intercepted_without_search_tool(self):
        self.assertIsNone(maybe_intercept_anthropic(_anthropic_request(tools=None), _Provider()))

    def test_non_streaming_loop(self):
        provider = _Provider(responses=[
            _msg([{"type": "text", "text": "Let me search."}, _tool_use("weather")], "tool_use"),
            _msg([{"type": "text", "text": "It is sunny."}], "end_turn"),
        ])
        loop = maybe_intercept_anthropic(_anthropic_request(), provider)
        self.assertEqual(loop.request.tools[-1]["name"], tools.INTERNAL_TOOL_NAME)
        result = run(loop.run())

        self.assertEqual(self._searches, [["weather"]])
        types = [b["type"] for b in result["content"]]
        self.assertEqual(types, ["text", "server_tool_use", "web_search_tool_result", "text"])
        self.assertEqual(result["stop_reason"], "end_turn")
        self.assertEqual(result["usage"]["input_tokens"], 20)
        self.assertEqual(result["usage"]["output_tokens"], 10)
        self.assertEqual(result["usage"]["server_tool_use"], {"web_search_requests": 1})
        self.assertEqual(loop.stats.header_value(), "rounds=1;queries=1")
        # Follow-up carries assistant tool_use + user tool_result.
        follow_up = provider.requests[1].messages
        self.assertEqual(follow_up[-2].role, "assistant")
        self.assertEqual(follow_up[-1].content[0].type, "tool_result")

    def test_mixed_tool_calls_return_client_tools(self):
        client_call = {"type": "tool_use", "id": "toolu_c", "name": "get_weather", "input": {}}
        provider = _Provider(responses=[_msg([_tool_use("q"), client_call], "tool_use")])
        result = run(maybe_intercept_anthropic(_anthropic_request(), provider).run())
        self.assertEqual(result["content"], [client_call])
        self.assertEqual(result["stop_reason"], "tool_use")
        self.assertEqual(self._searches, [])

    def test_limit_reached_nudges_then_ends(self):
        provider = _Provider(responses=[
            _msg([_tool_use("a")], "tool_use"),
            _msg([_tool_use("b")], "tool_use"),
            _msg([_tool_use("c")], "tool_use"),
        ])
        result = run(maybe_intercept_anthropic(_anthropic_request(), provider).run())
        self.assertEqual(self._searches, [["a"], ["b"]])
        last_result = provider.requests[2].messages[-1].content[0].content
        self.assertTrue(last_result.endswith(LIMIT_NUDGE))
        self.assertEqual(result["stop_reason"], "end_turn")
        self.assertEqual(result["content"][-1], {"type": "text", "text": LIMIT_REACHED_TEXT})

    def test_duplicate_query_stops(self):
        provider = _Provider(responses=[
            _msg([_tool_use("same")], "tool_use"),
            _msg([{"type": "text", "text": "partial"}, _tool_use("same")], "tool_use"),
        ])
        result = run(maybe_intercept_anthropic(_anthropic_request(), provider).run())
        self.assertEqual(self._searches, [["same"]])
        self.assertEqual(result["content"][-1], {"type": "text", "text": "partial"})

    def test_forced_tool_choice_is_relaxed_after_first_round(self):
        provider = _Provider(responses=[
            _msg([_tool_use("a")], "tool_use"),
            _msg([{"type": "text", "text": "done"}], "end_turn"),
        ])
        loop = maybe_intercept_anthropic(_anthropic_request(tool_choice={"type": "tool", "name": "web_search"}), provider)
        run(loop.run())
        first_tc = provider.requests[0].tool_choice
        first_tc = first_tc.model_dump() if hasattr(first_tc, "model_dump") else first_tc
        self.assertEqual(first_tc["name"], tools.INTERNAL_TOOL_NAME)
        self.assertEqual(provider.requests[1].tool_choice, {"type": "auto"})

    def test_streaming_loop(self):
        provider = _Provider(streams=[
            _stream_round([
                {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                {"type": "text", "text": "Searching. "},
                _tool_use("weather"),
            ], "tool_use"),
            _stream_round([{"type": "text", "text": "Sunny."}], "end_turn", input_tokens=30, output_tokens=7),
        ])
        loop = maybe_intercept_anthropic(_anthropic_request(stream=True), provider)
        chunks = run(_collect(loop.stream()))
        events = [(e, p) for c in chunks for e, p, _ in iter_sse_events(c)]
        names = [e for e, _ in events]
        self.assertEqual(names.count("message_start"), 1)
        self.assertEqual(names.count("message_stop"), 1)
        starts = [p for e, p in events if e == "content_block_start"]
        self.assertEqual([p["index"] for p in starts], [0, 1, 2, 3, 4])
        self.assertEqual(
            [p["content_block"]["type"] for p in starts],
            ["thinking", "text", "server_tool_use", "web_search_tool_result", "text"],
        )
        # The internal tool_use never reaches the client.
        self.assertNotIn(tools.INTERNAL_TOOL_NAME, "".join(chunks))
        delta = [p for e, p in events if e == "message_delta"][0]
        self.assertEqual(delta["delta"]["stop_reason"], "end_turn")
        self.assertEqual(delta["usage"]["input_tokens"], 40)
        self.assertEqual(delta["usage"]["output_tokens"], 12)
        # Follow-up reproduces the thinking block with its signature.
        assistant = provider.requests[1].messages[-2]
        self.assertEqual(assistant.content[0].type, "thinking")
        self.assertEqual(assistant.content[0].signature, "sig")
        self.assertEqual(assistant.content[2].input, {"query": "weather"})

    def test_streaming_max_tokens_mid_tool_use(self):
        provider = _Provider(streams=[_stream_round([{"type": "text", "text": "x"}, _tool_use("q")], "max_tokens")])
        chunks = run(_collect(maybe_intercept_anthropic(_anthropic_request(stream=True), provider).stream()))
        delta = [p for c in chunks for e, p, _ in iter_sse_events(c) if e == "message_delta"][0]
        self.assertEqual(delta["delta"]["stop_reason"], "max_tokens")
        self.assertEqual(self._searches, [])

    def test_replayed_history_is_rehydrated(self):
        native = tools.build_native_blocks("srvtoolu_wspz", fake_search_outcome("old"))
        request = _anthropic_request(messages=[
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": [*native, {"type": "text", "text": "a1"}]},
            {"role": "user", "content": "q2"},
        ])
        loop = maybe_intercept_anthropic(request, _Provider())
        roles = [m.role for m in loop.request.messages]
        self.assertEqual(roles, ["user", "assistant", "user", "assistant", "user"])
        self.assertEqual(loop.request.messages[1].content[0].type, "tool_use")

    def test_count_tokens_payload_rewritten(self):
        payload = {"model": "custom:test/claude", "messages": [], "tools": [{"type": "web_search_20250305", "name": "web_search"}]}
        out = rewrite_anthropic_count_tokens_payload(payload, _Provider())
        self.assertEqual(out["tools"][0]["name"], tools.INTERNAL_TOOL_NAME)


# ==================== Chat loop ====================

def _chat_response(message, finish, usage=(10, 5)):
    return ChatCompletionResponse(**{
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)},
    })


def _chat_call(query, cid="call_1", name=tools.INTERNAL_TOOL_NAME):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps({"query": query})}}


def _chat_request(**extra):
    body = {"model": "custom:test/gpt", "messages": [{"role": "user", "content": "news?"}], "web_search_options": {}}
    body.update(extra)
    return ChatCompletionRequest.model_validate(body)


def _chat_chunk(delta=None, finish=None, usage=None):
    obj = {"id": "chatcmpl-r", "object": "chat.completion.chunk", "created": 2, "model": "m",
           "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}] if usage is None else []}
    if usage is not None:
        obj["usage"] = usage
    return f"data: {json.dumps(obj)}\n\n"


class ChatLoopTests(_ConfigMixin, unittest.TestCase):
    def _loop(self, responses=None, streams=None, request=None):
        calls = []
        responses = list(responses or [])
        streams = list(streams or [])

        async def call(req):
            calls.append(req)
            return responses.pop(0)

        async def call_stream(req):
            calls.append(req)
            for chunk in streams.pop(0):
                yield chunk

        loop = maybe_intercept_chat(request or _chat_request(), "custom:test", call, call_stream)
        return loop, calls

    def test_non_streaming(self):
        loop, calls = self._loop(responses=[
            _chat_response({"content": None, "tool_calls": [_chat_call("weather")]}, "tool_calls"),
            _chat_response({"content": "Sunny."}, "stop"),
        ])
        result = run(loop.run())
        self.assertEqual(self._searches, [["weather"]])
        data = result.model_dump(exclude_unset=True)
        self.assertEqual(data["choices"][0]["message"]["content"], "Sunny.")
        self.assertNotIn("tool_calls", data["choices"][0]["message"])
        self.assertEqual(data["usage"]["prompt_tokens"], 20)
        self.assertNotIn("web_search_options", calls[0].model_dump())
        follow = calls[1].model_dump(exclude_none=True)["messages"]
        self.assertEqual(follow[-1]["role"], "tool")
        self.assertEqual(follow[-2]["tool_calls"][0]["function"]["name"], tools.INTERNAL_TOOL_NAME)

    def test_mixed_calls(self):
        loop, _ = self._loop(responses=[
            _chat_response({"content": None, "tool_calls": [_chat_call("q"), _chat_call("", "call_2", "get_weather")]}, "tool_calls"),
        ])
        data = run(loop.run()).model_dump(exclude_unset=True)
        calls = data["choices"][0]["message"]["tool_calls"]
        self.assertEqual([c["function"]["name"] for c in calls], ["get_weather"])
        self.assertEqual(data["choices"][0]["finish_reason"], "tool_calls")

    def test_streaming(self):
        round1 = [
            _chat_chunk({"role": "assistant", "content": "Let me check. "}),
            _chat_chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": tools.INTERNAL_TOOL_NAME, "arguments": ""}}]}),
            _chat_chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{\"query\": \"weather\"}"}}]}),
            _chat_chunk({}, "tool_calls"),
            _chat_chunk(usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
            "data: [DONE]\n\n",
        ]
        round2 = [
            _chat_chunk({"role": "assistant", "content": "Sunny."}),
            _chat_chunk({}, "stop"),
            _chat_chunk(usage={"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23}),
            "data: [DONE]\n\n",
        ]
        loop, calls = self._loop(streams=[round1, round2], request=_chat_request(stream=True))
        chunks = run(_collect(loop.stream()))
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        self.assertEqual(sum(c == "data: [DONE]\n\n" for c in chunks), 1)
        bodies = [json.loads(c[6:]) for c in chunks[:-1]]
        text = "".join((b["choices"][0]["delta"].get("content") or "") for b in bodies if b["choices"])
        self.assertEqual(text, "Let me check. Sunny.")
        self.assertNotIn(tools.INTERNAL_TOOL_NAME, "".join(chunks))
        finishes = [b["choices"][0]["finish_reason"] for b in bodies if b["choices"] and b["choices"][0]["finish_reason"]]
        self.assertEqual(finishes, ["stop"])
        self.assertEqual(bodies[-1]["usage"]["prompt_tokens"], 30)
        self.assertTrue(all(b["id"] == "chatcmpl-r" for b in bodies))
        self.assertEqual(self._searches, [["weather"]])
        self.assertEqual(calls[1].model_dump(exclude_none=True)["messages"][-2]["content"], "Let me check. ")


# ==================== Responses loop ====================

def _resp(output, rid="resp_1", usage=(10, 5)):
    return ResponseObject(**{
        "id": rid, "object": "response", "status": "completed", "model": "m", "output": output,
        "usage": {"input_tokens": usage[0], "output_tokens": usage[1], "total_tokens": sum(usage)},
    })


def _fc(query, call_id="fc_1"):
    return {"type": "function_call", "id": "item_" + call_id, "call_id": call_id, "name": tools.INTERNAL_TOOL_NAME,
            "arguments": json.dumps({"query": query}), "status": "completed"}


def _message_out(text):
    return {"type": "message", "id": "msg_x", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


class ResponsesLoopTests(_ConfigMixin, unittest.TestCase):
    def _loop(self, responses, **extra):
        calls = []
        responses = list(responses)

        async def call(req):
            calls.append(req)
            return responses.pop(0)

        body = {"model": "custom:test/gpt", "input": "news?", "tools": [{"type": "web_search_preview"}]}
        body.update(extra)
        loop = maybe_intercept_responses(ResponsesCreateRequest.model_validate(body), "custom:test", call)
        return loop, calls

    def test_previous_response_id_follow_up(self):
        loop, calls = self._loop([_resp([_fc("weather")]), _resp([_message_out("Sunny.")], rid="resp_2")])
        result = run(loop.run()).model_dump(exclude_none=True)
        self.assertEqual([i["type"] for i in result["output"]], ["web_search_call", "message"])
        self.assertEqual(result["output_text"], "Sunny.")
        self.assertEqual(result["usage"]["input_tokens"], 20)
        self.assertEqual(result["tools"], [{"type": "web_search_preview"}])
        follow = calls[1].model_dump(exclude_none=True)
        self.assertEqual(follow["previous_response_id"], "resp_1")
        self.assertEqual(follow["input"][0]["type"], "function_call_output")

    def test_stateless_follow_up_replays_input(self):
        loop, calls = self._loop([_resp([_fc("weather")]), _resp([_message_out("Sunny.")])], store=False)
        run(loop.run())
        follow = calls[1].model_dump(exclude_none=True)
        self.assertNotIn("previous_response_id", follow)
        self.assertEqual([i.get("type", "message") for i in follow["input"]], ["message", "function_call", "function_call_output"])

    def test_synthesized_stream(self):
        loop, _ = self._loop([_resp([_fc("weather")]), _resp([_message_out("Sunny.")])])
        data = run(loop.run()).model_dump(exclude_none=True)
        events = [e for c in synthesize_responses_stream(data) for e, _, _ in iter_sse_events(c)]
        self.assertEqual(events[0], "response.created")
        self.assertEqual(events[-1], "response.completed")
        self.assertIn("response.output_text.delta", events)

    def _client_fc(self, call_id="fc_c"):
        return {"type": "function_call", "id": "item_" + call_id, "call_id": call_id, "name": "get_weather",
                "arguments": "{}", "status": "completed"}

    def _follow_up(self, previous_response_id, input_value):
        return settle_pending_responses_calls(ResponsesCreateRequest.model_validate({
            "model": "custom:test/gpt", "previous_response_id": previous_response_id, "input": input_value,
        }))

    def test_mixed_exit_settles_internal_call_next_turn(self):
        loop, _ = self._loop([_resp([_fc("q", "fc_int"), self._client_fc()], rid="resp_mixed")])
        result = run(loop.run()).model_dump(exclude_none=True)
        self.assertEqual([i.get("name") for i in result["output"]], ["get_weather"])
        client_output = {"type": "function_call_output", "call_id": "fc_c", "output": "72F"}
        follow = self._follow_up("resp_mixed", [client_output]).model_dump(exclude_none=True)
        self.assertEqual([i["call_id"] for i in follow["input"]], ["fc_int", "fc_c"])
        self.assertEqual(follow["input"][0]["type"], "function_call_output")
        # A retried turn is settled again; an unrelated chain is untouched.
        self.assertEqual(len(self._follow_up("resp_mixed", [client_output]).input), 2)
        self.assertEqual(self._follow_up("resp_other", [client_output]).input, [client_output])

    def test_limit_exit_settles_internal_call_next_turn(self):
        loop, _ = self._loop([
            _resp([_fc("a")], rid="r1"), _resp([_fc("b")], rid="r2"), _resp([_fc("c", "fc_3")], rid="r3"),
        ])
        result = run(loop.run()).model_dump(exclude_none=True)
        self.assertEqual(result["output_text"], LIMIT_REACHED_TEXT)
        follow = self._follow_up("r3", "thanks").model_dump(exclude_none=True)
        self.assertEqual(follow["input"][0]["call_id"], "fc_3")
        self.assertEqual(follow["input"][1], {"role": "user", "content": "thanks"})

    def test_stateless_mixed_exit_records_nothing(self):
        loop, _ = self._loop([_resp([_fc("q", "fc_int"), self._client_fc()], rid="resp_sl")], store=False)
        run(loop.run())
        self.assertEqual(self._follow_up("resp_sl", "hi").input, "hi")

    def test_stream_reports_progress_per_round(self):
        loop, _ = self._loop([_resp([_fc("weather")]), _resp([_message_out("Sunny.")], rid="resp_2")])
        chunks = run(_collect(responses_websearch_stream(loop)))
        self.assertEqual(chunks[0], ": websearch round completed\n\n")
        events = [e for c in chunks[1:] for e, _, _ in iter_sse_events(c)]
        self.assertEqual(events[0], "response.created")
        self.assertEqual(events[-1], "response.completed")

    def _stream_error(self, exc):
        async def call(req):
            raise exc

        body = {"model": "custom:test/gpt", "input": "news?", "tools": [{"type": "web_search_preview"}]}
        loop = maybe_intercept_responses(ResponsesCreateRequest.model_validate(body), "custom:test", call)
        chunks = run(_collect(responses_websearch_stream(loop)))
        self.assertEqual(len(chunks), 1)
        return json.loads(chunks[0].split("data: ", 1)[1])

    def test_stream_maps_provider_http_error(self):
        err = self._stream_error(ProviderHTTPError(429, "rate", {"error": {"code": "rate_limit_exceeded", "message": "slow down"}}))
        self.assertEqual((err["code"], err["message"]), ("rate_limit_exceeded", "slow down"))
        err = self._stream_error(ProviderHTTPError(400, "bad", {"error": "plain string"}))
        self.assertEqual((err["code"], err["message"]), ("400", "plain string"))
        err = self._stream_error(ValueError("bad input"))
        self.assertEqual((err["code"], err["message"]), ("invalid_request_error", "bad input"))


# ==================== Route integration (/v1/messages) ====================

import tests.test_anthropic_messages as _tam  # module import: not re-collected
from app.providers.custom_providers import CustomProvider
from app.request_tracker import request_tracker


class _ScriptedCustomProvider(CustomProvider):
    def __init__(self, responses=None, streams=None):
        self.full_provider_name = "custom:test"
        self._supported_apis = ["anthropic"]
        self.responses = list(responses or [])
        self.streams = list(streams or [])
        self.requests = []

    async def anthropic_messages(self, request, anthropic_beta=None):
        self.requests.append(request)
        return self.responses.pop(0)

    async def anthropic_messages_stream(self, request, anthropic_beta=None):
        self.requests.append(request)
        for chunk in self.streams.pop(0):
            yield chunk


class AnthropicRouteTests(_ConfigMixin, unittest.TestCase):
    _invoke_create_message = _tam.AnthropicMessagesRouteTests._invoke_create_message
    _make_request_obj = staticmethod(_tam.AnthropicMessagesRouteTests._make_request_obj)
    _start_tracking_request = _tam.AnthropicMessagesRouteTests._start_tracking_request
    _collect_response_body = staticmethod(_tam.AnthropicMessagesRouteTests._collect_response_body)

    def setUp(self):
        super().setUp()
        import itertools
        asyncio.run(request_tracker.stop())
        asyncio.run(request_tracker.start())
        self._request_ids = itertools.count(1)

    def tearDown(self):
        asyncio.run(request_tracker.stop())
        super().tearDown()

    def _payload(self, **extra):
        payload = {
            "model": "custom:test/claude",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "news?"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        }
        payload.update(extra)
        return payload

    def test_non_streaming_route(self):
        provider = _ScriptedCustomProvider(responses=[
            _msg([_tool_use("weather")], "tool_use"),
            _msg([{"type": "text", "text": "Sunny."}], "end_turn"),
        ])
        response, body, _ = self._invoke_create_message(self._payload(stream=False), provider)
        self.assertEqual(response.status_code, 200)
        data = json.loads(body)
        self.assertEqual([b["type"] for b in data["content"]], ["server_tool_use", "web_search_tool_result", "text"])
        self.assertEqual(response.headers["x-llmproxy-websearch"], "rounds=1;queries=1")
        self.assertNotIn("x-llmproxy-dropped-anthropic-fields", response.headers)
        # The native server tool never reaches the provider.
        sent_tools = provider.requests[0].tools
        self.assertEqual([t["name"] for t in sent_tools], [tools.INTERNAL_TOOL_NAME])

    def test_streaming_route(self):
        provider = _ScriptedCustomProvider(streams=[
            _stream_round([_tool_use("weather")], "tool_use"),
            _stream_round([{"type": "text", "text": "Sunny."}], "end_turn"),
        ])
        response, body, _ = self._invoke_create_message(self._payload(stream=True), provider)
        self.assertEqual(response.headers["x-llmproxy-websearch"], "enabled")
        names = [e for e, _, _ in iter_sse_events(body)]
        self.assertEqual(names[0], "message_start")
        self.assertEqual(names[-1], "message_stop")
        self.assertIn("web_search_tool_result", body)
        self.assertEqual(self._searches, [["weather"]])

    def test_disabled_passes_through(self):
        websearch_settings_cache.set_config(CFG.with_overrides(enabled=False))
        provider = _ScriptedCustomProvider(responses=[_msg([{"type": "text", "text": "hi"}], "end_turn")])
        response, body, _ = self._invoke_create_message(self._payload(stream=False), provider)
        self.assertNotIn("x-llmproxy-websearch", response.headers)
        self.assertEqual(provider.requests[0].tools[0]["type"], "web_search_20250305")


# ==================== Admin settings ====================

class AdminSettingsTests(unittest.TestCase):
    def test_validation(self):
        from pydantic import ValidationError
        from app.auth.models import WebSearchSettingsUpdate

        with self.assertRaises(ValidationError):
            WebSearchSettingsUpdate(enabled=True)  # URL required when enabled
        with self.assertRaises(ValidationError):
            WebSearchSettingsUpdate(searxng_base_url="ftp://x")
        with self.assertRaises(ValidationError):
            WebSearchSettingsUpdate(apply_to=["embeddings"])
        with self.assertRaises(ValidationError):
            WebSearchSettingsUpdate(max_agentic_loops=0)
        ok = WebSearchSettingsUpdate(enabled=True, searxng_base_url=" http://s:8080/ ", engines="  ")
        self.assertEqual(ok.searxng_base_url, "http://s:8080")
        self.assertIsNone(ok.engines)

    def test_backend_validation(self):
        from pydantic import ValidationError
        from app.auth.models import WebSearchSettingsUpdate

        with self.assertRaises(ValidationError):
            WebSearchSettingsUpdate(provider="bing")
        with self.assertRaises(ValidationError):
            WebSearchSettingsUpdate(fourget_scraper="not_a_scraper")
        with self.assertRaises(ValidationError):
            WebSearchSettingsUpdate(fourget_base_url="ftp://x")
        with self.assertRaises(ValidationError):
            WebSearchSettingsUpdate(fourget_country="a bad value")
        # The *selected* backend's URL is the one that is required.
        with self.assertRaises(ValidationError):
            WebSearchSettingsUpdate(enabled=True, provider="fourget", searxng_base_url="http://s")
        ok = WebSearchSettingsUpdate(
            enabled=True, provider="fourget", fourget_base_url=" http://fourget/ ",
            fourget_scraper="ddg", fourget_lang="en", fourget_country="us-en",
        )
        self.assertEqual(ok.fourget_base_url, "http://fourget")
        self.assertIsNone(ok.searxng_base_url)

    def test_row_round_trip(self):
        from types import SimpleNamespace
        from app.auth.models import FOURGET_WEB_SCRAPERS, WebSearchSettingsUpdate
        from app.routes.admin import _websearch_response, _websearch_values
        from app.websearch.settings import config_from_row

        self.assertEqual(_websearch_values(WebSearchSettingsUpdate(searxng_base_url="http://s"))["searxng_base_url"], "http://s")

        row = SimpleNamespace(
            enabled=True, searxng_base_url="http://s", engines=None,
            categories=None, language=None, safesearch=None, time_range=None, max_results=None,
            max_snippet_chars=None, timeout_seconds=None, max_agentic_loops=4, max_queries_per_turn=None,
            apply_to=json.dumps(["chat_completions"]), enabled_providers=json.dumps(["*"]),
            provider=None, fourget_base_url=None, fourget_scraper=None, fourget_lang=None, fourget_country=None,
            updated_at=None, updated_by="admin",
        )
        cfg = config_from_row(row)
        self.assertEqual(cfg.max_agentic_loops, 4)
        self.assertEqual(cfg.max_results, 5)
        self.assertEqual(cfg.apply_to, frozenset({"chat_completions"}))
        # A row predating the backend columns (NULL after the migration) defaults to SearXNG.
        self.assertEqual(cfg.provider, "searxng")
        self.assertIsNone(cfg.fourget_base_url)
        self.assertEqual(cfg.fourget_scraper, FOURGET_WEB_SCRAPERS[0])
        self.assertEqual(cfg.base_url, "http://s")
        response = _websearch_response(row)
        self.assertEqual(response.searxng_base_url, "http://s")
        self.assertIsNone(config_from_row(None).searxng_base_url)

    def test_fourget_row_round_trip(self):
        from types import SimpleNamespace
        from app.auth.models import WebSearchSettingsUpdate
        from app.routes.admin import _websearch_response, _websearch_values
        from app.websearch.settings import config_from_row

        values = _websearch_values(WebSearchSettingsUpdate(
            enabled=True, provider="fourget", fourget_base_url="http://fourget",
            fourget_scraper="brave", fourget_lang="en", fourget_country="us",
        ))
        # Every key must be both a column name and a WebSearchConfig field: the
        # same dict is handed to upsert_websearch_settings and to with_overrides.
        for key in ("provider", "fourget_base_url", "fourget_scraper", "fourget_lang", "fourget_country"):
            self.assertIn(key, values)

        row = SimpleNamespace(
            enabled=True, provider="fourget", searxng_base_url=None, engines=None,
            categories=None, language=None, safesearch=None, time_range=None,
            fourget_base_url="http://fourget", fourget_scraper="brave",
            fourget_lang="en", fourget_country="us",
            max_results=None, max_snippet_chars=None, timeout_seconds=None,
            max_agentic_loops=None, max_queries_per_turn=None,
            apply_to=None, enabled_providers=json.dumps(["*"]),
            updated_at=None, updated_by="admin",
        )
        cfg = config_from_row(row)
        self.assertEqual(cfg.base_url, "http://fourget")
        self.assertEqual(cfg.fourget_scraper, "brave")
        self.assertEqual(_websearch_response(row).provider, "fourget")

    def test_is_active(self):
        from app.websearch.settings import WebSearchSettingsCache

        cache = WebSearchSettingsCache()
        self.assertFalse(cache.is_active("anthropic_messages", "custom:test"))
        cache.set_config(CFG)
        self.assertTrue(cache.is_active("anthropic_messages", "custom:test"))
        # The gate follows the selected backend's URL, not SearXNG's.
        cache.set_config(CFG.with_overrides(provider="fourget"))
        self.assertFalse(cache.is_active("anthropic_messages", "custom:test"))
        cache.set_config(CFG.with_overrides(provider="fourget", fourget_base_url="http://fourget"))
        self.assertTrue(cache.is_active("anthropic_messages", "custom:test"))
        cache.set_config(CFG)
        self.assertFalse(cache.is_active("anthropic_messages", "bedrock:x"))
        cache.set_config(CFG.with_overrides(enabled_providers=frozenset({"*"}), apply_to=frozenset({"responses"})))
        self.assertTrue(cache.is_active("responses", "bedrock:x"))
        self.assertFalse(cache.is_active("chat_completions", "bedrock:x"))


if __name__ == "__main__":
    unittest.main()
