"""PlatformAI provider end to end against a stub v3 gateway (httpx.MockTransport)."""

import asyncio
import json
import unittest

import httpx

from app.anthropic_models import AnthropicMessagesRequest
from app.openai_models import ChatCompletionRequest, CompletionRequest, EmbeddingRequest
from app.providers.base import ProviderHTTPError
from app.providers.platformai import PlatformAIProvider, limits
from app.providers.platformai.auth import reset_token_cache

MODEL = "platformai:main/claude-haiku-5-5"
# A model with no known unsupported params (no model_capabilities rule, no
# gateway default), for tests that learn them.
OPEN_MODEL = "platformai:main/claude-sonnet-4-5"
KEY_PATH = "/pai/key_auth/v3/ai/chat"
OAUTH_PATH = "/pai/api/v3/ai/chat"


def _sse(*frames, tail=b""):
    return "".join(f"event: generate\ndata: {json.dumps(f)}\n\n" for f in frames).encode() + tail


def _text(t, **extra):
    return {"type": "text", "text": t, **extra}


def _reply(text, finish="stop", **usage):
    return {"data": {"model": "claude-haiku-5-5",
                     "usage": usage or {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                     "messages": [{"role": "assistant", "finish_reason": finish, "contents": [_text(text)]}]}}


def _refusal(code, message):
    return httpx.Response(400, json={"error": {"code": code, "message": message}})


class StubGateway:
    def __init__(self):
        self.requests = []
        self.chat_responses = []  # queue of httpx.Response

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/oauth/v2/token":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        body = json.loads(request.content) if request.content else None
        self.requests.append((path, body, request))
        if path in (KEY_PATH, OAUTH_PATH):
            return self.chat_responses.pop(0)
        return httpx.Response(404, json={"error": {"code": "API-url_not_found", "message": "bad url"}})

    def bodies(self):
        return [b for _, b, _ in self.requests]


def _provider(stub, **config):
    config = {"name": "main", "provider_name": "platformai", "provider_key": "platformai:main",
              "api_key": "193-key", "models": ["claude-haiku-5-5", "gpt-5.5"], **config}
    provider = PlatformAIProvider(config)
    client = httpx.AsyncClient(transport=httpx.MockTransport(stub))
    provider.client = client
    if provider.tokens is not None:
        provider.tokens._http = client
    provider.gateway.http = client
    return provider


def _collect(agen):
    async def run():
        return [c async for c in agen]
    return asyncio.run(run())


def _sse_payloads(chunks):
    out = []
    for c in chunks:
        for line in c.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                out.append(json.loads(line[6:]))
    return out


def _chat(**kwargs):
    kwargs.setdefault("model", MODEL)
    kwargs.setdefault("messages", [{"role": "user", "content": "hello"}])
    return ChatCompletionRequest(**kwargs)


class ProviderTests(unittest.TestCase):
    def setUp(self):
        reset_token_cache()
        limits.reset_limits()
        self.addCleanup(reset_token_cache)
        self.addCleanup(limits.reset_limits)
        self.stub = StubGateway()
        self.provider = _provider(self.stub)

    def test_identity_capabilities_and_models(self):
        self.assertEqual(self.provider.provider_type, "platformai")
        self.assertEqual(self.provider.base_url, "https://platformai.zoho.in")
        self.assertEqual(self.provider.get_supported_apis(), ["openai", "anthropic"])
        self.assertNotIn("/v1/embeddings", self.provider.get_supported_endpoints())
        self.assertEqual(self.provider.get_anthropic_mode_for_model(MODEL), "adapter")
        models = asyncio.run(self.provider.get_available_models())
        self.assertEqual([m.id for m in models], ["platformai:main/claude-haiku-5-5", "platformai:main/gpt-5.5"])
        self.assertEqual(self.stub.requests, [])  # no network call
        with self.assertRaises(NotImplementedError):
            asyncio.run(self.provider.embeddings(EmbeddingRequest(model=MODEL, input="x")))

    def test_config_requires_credentials(self):
        with self.assertRaises(ValueError):
            PlatformAIProvider({"name": "main", "provider_name": "platformai"})

    def test_api_key_request(self):
        self.stub.chat_responses.append(httpx.Response(200, json=_reply("Hi!")))
        resp = asyncio.run(self.provider.chat_completion(_chat(
            reasoning_effort="high", metadata={"custom_param_a": "chat1"})))
        self.assertEqual(resp.choices[0].message.content, "Hi!")
        self.assertEqual(resp.choices[0].finish_reason, "stop")
        self.assertEqual((resp.usage.prompt_tokens, resp.usage.completion_tokens), (5, 2))
        path, body, request = self.stub.requests[0]
        self.assertEqual(path, KEY_PATH)
        self.assertEqual(request.headers["authorization"], "Bearer 193-key")
        self.assertNotIn("z-token-id", request.headers)
        self.assertEqual(json.loads(request.headers["z-pai-metadata"])["custom_param_a"], "chat1")
        self.assertEqual(body, {"model": "claude-haiku-5-5", "messages": [{"role": "user", "contents": [_text("hello")]}],
                                "stream": False, "reasoning": {"effort": "high"},
                                "prompt_caching": {"name": "ephemeral", "ttl": "1h"}})

    def test_oauth_request_and_byok_token(self):
        provider = _provider(self.stub, api_key=None, oauth_client_id="cid", oauth_refresh_token="1000.r",
                             pai_token_id="5434566543234")
        self.assertEqual(provider.auth_mode, "oauth")
        self.stub.chat_responses.append(httpx.Response(200, json=_reply("ok")))
        asyncio.run(provider.chat_completion(_chat()))
        path, _, request = self.stub.requests[0]
        self.assertEqual(path, OAUTH_PATH)
        self.assertEqual(request.headers["authorization"], "Zoho-oauthtoken tok")
        self.assertEqual(request.headers["z-token-id"], "5434566543234")

    def test_variations_become_choices(self):
        self.stub.chat_responses.append(httpx.Response(200, json={"data": {"messages": [
            {"role": "assistant", "index": 0, "finish_reason": "stop", "contents": [_text("a")]},
            {"role": "assistant", "index": 1, "finish_reason": "stop", "contents": [_text("b")]},
        ]}}))
        resp = asyncio.run(self.provider.chat_completion(_chat(model=OPEN_MODEL, n=2)))
        self.assertEqual([c.message.content for c in resp.choices], ["a", "b"])
        self.assertEqual(self.stub.bodies()[0]["variations"], 2)

    def test_tool_call_response(self):
        self.stub.chat_responses.append(httpx.Response(200, json={"data": {"messages": [{
            "role": "assistant", "finish_reason": "tool_calls", "tools": [{
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Chennai\"}"},
                "id": "toolu_01", "type": "function"}]}]}}))
        resp = asyncio.run(self.provider.chat_completion(_chat(tools=[
            {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}])))
        message = resp.choices[0].message
        self.assertIsNone(message.content)
        self.assertEqual(message.tool_calls[0].id, "toolu_01")
        self.assertEqual(message.tool_calls[0].function.arguments, '{"city":"Chennai"}')
        self.assertEqual(resp.choices[0].finish_reason, "tool_calls")

    def test_unsupported_param_is_dropped_and_remembered(self):
        self.stub.chat_responses += [
            _refusal("PAI-MODEL-002", "temperature not supported for the model claude-sonnet-4-5"),
            httpx.Response(200, json=_reply("ok")), httpx.Response(200, json=_reply("ok")),
        ]
        req = _chat(model=OPEN_MODEL, temperature=0.3, top_p=0.5)
        asyncio.run(self.provider.chat_completion(req))
        asyncio.run(self.provider.chat_completion(req))
        self.assertEqual(["temperature" in b for b in self.stub.bodies()], [True, False, False])
        self.assertTrue(all(b["top_p"] == 0.5 for b in self.stub.bodies()))

    def test_unsupported_response_format_falls_back_to_instruction(self):
        self.stub.chat_responses += [
            _refusal("PAI-MODEL-002", "response_format not supported for the model claude-sonnet-4-5"),
            httpx.Response(200, json=_reply('{"a":1}')),
        ]
        asyncio.run(self.provider.chat_completion(_chat(model=OPEN_MODEL, response_format={"type": "json_object"})))
        retry = self.stub.bodies()[1]
        self.assertNotIn("response_format", retry)
        self.assertIn("single valid JSON object", retry["context"][0]["text"])

    def test_unsupported_variations_is_a_client_error(self):
        self.stub.chat_responses.append(
            _refusal("PAI-MODEL-002", "variations not supported for the model claude-sonnet-4-5"))
        with self.assertRaises(ProviderHTTPError) as ctx:
            asyncio.run(self.provider.chat_completion(_chat(model=OPEN_MODEL, n=2)))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(len(self.stub.bodies()), 1)

    def test_default_unsupported_params_need_no_refusal(self):
        self.stub.chat_responses.append(httpx.Response(200, json=_reply('{"a":1}')))
        asyncio.run(self.provider.chat_completion(_chat(
            temperature=0.3, top_p=0.5, top_k=4, seed=1, response_format={"type": "json_object"})))
        (body,) = self.stub.bodies()
        for param in ("temperature", "top_p", "top_k", "response_format"):
            self.assertNotIn(param, body)
        self.assertEqual(body["seed"], 1)
        self.assertIn("single valid JSON object", body["context"][0]["text"])
        with self.assertRaises(ProviderHTTPError) as ctx:
            asyncio.run(self.provider.chat_completion(_chat(n=2)))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(len(self.stub.bodies()), 1)  # rejected locally
        limits.reset_limits()
        # Sampling params come from app.model_capabilities; only gateway refusals live in limits.
        self.assertEqual(limits.unsupported_params("platformai:main", "claude-haiku-5-5"),
                         {"response_format", "variations"})

    def test_max_tokens_refusal_learns_ceiling(self):
        self.stub.chat_responses += [
            _refusal("PAI-PROVIDER-002", 'Bad request to the provider: {"type":"invalid_request_error","message":'
                     '"max_tokens: 10000000 > 128000, which is the maximum allowed number of output tokens for m"}'),
            httpx.Response(200, json=_reply("ok")),
        ]
        asyncio.run(self.provider.chat_completion(_chat(max_tokens=10_000_000)))
        self.assertEqual([b["max_tokens"] for b in self.stub.bodies()], [10_000_000, 128000])

    def test_effort_refusal_steps_down(self):
        self.stub.chat_responses += [
            _refusal("PAI-PROVIDER-002", "Bad request to the provider: This model does not support effort level 'high'."),
            httpx.Response(200, json=_reply("ok")),
        ]
        asyncio.run(self.provider.chat_completion(_chat(reasoning_effort="high")))
        self.assertEqual([b["reasoning"] for b in self.stub.bodies()], [{"effort": "high"}, {"effort": "medium"}])

    def test_other_400_not_retried(self):
        self.stub.chat_responses.append(_refusal("API-json_extra_key", "extra key found in JSON: 'x'"))
        with self.assertRaises(ProviderHTTPError) as ctx:
            asyncio.run(self.provider.chat_completion(_chat()))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(len(self.stub.bodies()), 1)

    def test_unknown_model_is_404(self):
        self.stub.chat_responses.append(_refusal("PAI-TOKEN-003", "API Key not found"))
        with self.assertRaises(ProviderHTTPError) as ctx:
            asyncio.run(self.provider.chat_completion(_chat(model="platformai:main/nope")))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_streaming_text_and_usage(self):
        self.stub.chat_responses.append(httpx.Response(200, content=_sse(
            {"usage": {"completion_tokens": 2, "prompt_tokens": 16, "total_tokens": 18},
             "messages": [{"role": "assistant", "contents": [_text("")]}], "model": "claude-haiku-5-5"},
            {"messages": [{}], "model": "claude-haiku-5-5"},
            {"messages": [{"contents": [_text("1, 2")]}], "model": "claude-haiku-5-5"},
            {"messages": [{"contents": [_text(", 3")]}], "model": "claude-haiku-5-5"},
            {"is_final_chunk": True, "usage": {"completion_tokens": 9, "prompt_tokens": 16, "total_tokens": 25},
             "messages": [{"contents": [_text("1, 2, 3")], "index": 0}], "model": "claude-haiku-5-5"},
            {"messages": [{"contents": [_text("after final")]}]},
        )))
        chunks = _collect(self.provider.chat_completion_stream(_chat(stream=True)))
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        self.assertEqual(sum(c == "data: [DONE]\n\n" for c in chunks), 1)
        payloads = _sse_payloads(chunks)
        text = "".join(p["choices"][0]["delta"].get("content") or "" for p in payloads if p.get("choices"))
        self.assertEqual(text, "1, 2, 3")
        self.assertEqual([p["choices"][0]["finish_reason"] for p in payloads if p.get("choices")][-1], "stop")
        self.assertEqual(payloads[-1]["usage"]["completion_tokens"], 9)
        self.assertIs(self.stub.bodies()[0]["stream"], True)

    def test_streaming_tool_call_survives_gateway_error_event(self):
        self.stub.chat_responses.append(httpx.Response(200, content=_sse(
            {"messages": [{"role": "assistant", "contents": [_text("")]}], "model": "m"},
            {"messages": [{"contents": [_text("Checking.")]}], "model": "m"},
            {"messages": [{"tools": [{"function": {"name": "get_weather", "arguments": ""},
                                      "id": "toolu_01", "type": "function"}]}], "model": "m"},
            {"messages": [{"tools": [{"function": {"arguments": "{\"city\":"}, "type": "function"}]}], "model": "m"},
            {"messages": [{"tools": [{"function": {"arguments": "\"Chennai\"}"}, "type": "function"}]}], "model": "m"},
            tail=b'event: error\ndata: {"error":{"code":500,"requestURI":"/pai/key_auth/v3/ai/chat",'
                 b'"message":"Something went wrong"}}\n\n',
        )))
        chunks = _collect(self.provider.chat_completion_stream(_chat(stream=True, tools=[
            {"type": "function", "function": {"name": "get_weather"}}])))
        payloads = _sse_payloads(chunks)
        self.assertFalse(any("error" in p for p in payloads))
        tool_chunk = next(p for p in payloads if p.get("choices") and p["choices"][0]["delta"].get("tool_calls"))
        self.assertEqual(tool_chunk["choices"][0]["delta"]["tool_calls"][0]["function"],
                         {"name": "get_weather", "arguments": '{"city":"Chennai"}'})
        self.assertEqual([p["choices"][0]["finish_reason"] for p in payloads if p.get("choices")][-1], "tool_calls")

    def test_streaming_error_event_becomes_sse_error(self):
        self.stub.chat_responses.append(httpx.Response(200, content=_sse(
            {"messages": [{"contents": [_text("partial")]}]},
            tail=b'event: error\ndata: {"error": {"code": "PAI-PROVIDER-001", "message": "rate limited"}}\n\n',
        )))
        chunks = _collect(self.provider.chat_completion_stream(_chat(stream=True)))
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        errors = [p for p in _sse_payloads(chunks) if "error" in p]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["error"]["code"], 429)
        self.assertIn("rate limited", errors[0]["error"]["message"])
        self.assertFalse(any(p.get("choices") and p["choices"][0].get("finish_reason") for p in _sse_payloads(chunks)))

    def test_streaming_http_error_becomes_sse_error(self):
        self.stub.chat_responses.append(httpx.Response(429, text="slow down", headers={"retry-after": "3"}))
        chunks = _collect(self.provider.chat_completion_stream(_chat(stream=True)))
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        self.assertEqual(_sse_payloads(chunks)[0]["error"]["code"], 429)

    def test_completion_wraps_chat(self):
        self.stub.chat_responses.append(httpx.Response(200, json=_reply("4")))
        resp = asyncio.run(self.provider.completion(CompletionRequest(model=MODEL, prompt="2+2=")))
        self.assertEqual(resp.choices[0].text, "4")

    def test_anthropic_messages_via_adapter(self):
        self.stub.chat_responses.append(httpx.Response(200, json=_reply(
            "Hello from PlatformAI", prompt_tokens=9, completion_tokens=3, total_tokens=12, cached_tokens=4)))
        req = AnthropicMessagesRequest(model=OPEN_MODEL, max_tokens=256, system="be nice", top_k=5,
                                       thinking={"type": "enabled", "budget_tokens": 8000},
                                       messages=[{"role": "user", "content": "hi"}])
        result = asyncio.run(self.provider.anthropic_messages(req))
        self.assertEqual(result["content"][0]["text"], "Hello from PlatformAI")
        self.assertEqual(result["stop_reason"], "end_turn")
        self.assertEqual(result["usage"]["cache_read_input_tokens"], 4)
        self.assertEqual(result["usage"]["input_tokens"], 9)  # gateway prompt_tokens exclude cache reads
        body = self.stub.bodies()[0]
        self.assertEqual(body["context"], [_text("be nice")])
        self.assertEqual(body["max_tokens"], 256)
        self.assertEqual(body["top_k"], 5)
        self.assertEqual(body["reasoning"], {"effort": "high"})

    def test_anthropic_messages_stream_via_adapter(self):
        # The adapter estimates input tokens for message_start; keep tiktoken
        # from fetching its encoding over the network.
        from unittest import mock
        from app.conversion import token_estimate
        patcher = mock.patch.object(token_estimate, "_get_counter", return_value=token_estimate._heuristic_count)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.stub.chat_responses.append(httpx.Response(200, content=_sse(
            {"messages": [{"contents": [_text("Hi")]}]},
            {"is_final_chunk": True, "messages": [{"contents": [_text("Hi")]}],
             "usage": {"prompt_tokens": 2, "completion_tokens": 1}},
        )))
        req = AnthropicMessagesRequest(model=MODEL, max_tokens=64, stream=True,
                                       messages=[{"role": "user", "content": "hi"}])
        out = "".join(_collect(self.provider.anthropic_messages_stream(req)))
        self.assertIn("event: message_start", out)
        self.assertIn('"text":"Hi"', out.replace(" ", ""))
        self.assertIn("event: message_stop", out)


if __name__ == "__main__":
    unittest.main()
