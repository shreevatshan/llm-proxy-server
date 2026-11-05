"""PlatformAI: OpenAI chat -> v3 native chat body, schema sanitizer, usage/finish mapping."""

import unittest

from app.openai_models import ChatCompletionRequest
from app.providers.base import ProviderHTTPError
from app.providers.platformai.schema import sanitize_tool_schema
from app.providers.platformai.translate import (
    build_chat_body,
    build_usage,
    map_effort,
    merge_usage,
    normalize_finish_reason,
    step_down_effort,
)

PNG_URI = "data:image/png;base64,iVBORw0KGgo="


def _req(**kwargs) -> ChatCompletionRequest:
    kwargs.setdefault("model", "platformai:main/claude-haiku-5-5")
    kwargs.setdefault("messages", [{"role": "user", "content": "hi"}])
    return ChatCompletionRequest(**kwargs)


def _body(request=None, *, model="claude-haiku-5-5", stream=False, **kw):
    return build_chat_body(request or _req(), upstream_model=model, stream=stream, **kw)


def _text(t):
    return {"type": "text", "text": t}


class MessageTranslationTests(unittest.TestCase):
    def test_minimal_body(self):
        body = _body()
        self.assertEqual(body, {"model": "claude-haiku-5-5", "messages": [{"role": "user", "contents": [_text("hi")]}],
                                "stream": False, "prompt_caching": {"name": "ephemeral", "ttl": "1h"}})
        self.assertNotIn("prompt_caching", _body(model="gpt-5.5"))
        self.assertNotIn("prompt_caching", _body(unsupported={"prompt_caching"}))

    def test_system_and_developer_become_context_parts(self):
        body = _body(_req(messages=[
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "q"},
            {"role": "developer", "content": [{"type": "text", "text": "no emoji"}]},
        ]))
        self.assertEqual(body["context"], [_text("be terse"), _text("no emoji")])
        self.assertEqual(body["messages"], [{"role": "user", "contents": [_text("q")]}])

    def test_images(self):
        body = _body(_req(messages=[{"role": "user", "content": [
            {"type": "text", "text": "what?"},
            {"type": "image_url", "image_url": {"url": PNG_URI}},
            {"type": "image_url", "image_url": {"url": "https://x.test/a.jpg?s=1"}},
            {"type": "image_url", "image_url": {"url": "https://x.test/img"}},
            {"type": "file", "file": {"file_data": PNG_URI}},
        ]}]))
        self.assertEqual(body["messages"][0]["contents"], [
            _text("what?"),
            {"type": "image", "image": {"type": "base64", "mime_type": "image/png", "base64": PNG_URI}},
            {"type": "image", "image": {"type": "url", "url": "https://x.test/a.jpg?s=1", "mime_type": "image/jpeg"}},
            {"type": "image", "image": {"type": "url", "url": "https://x.test/img"}},
            {"type": "image", "image": {"type": "base64", "mime_type": "image/png", "base64": PNG_URI}},
        ])

    def test_unsupported_parts_rejected(self):
        for part in (
            {"type": "input_audio", "input_audio": {}},
            {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBE", "filename": "a.pdf"}},
            {"type": "image_url", "image_url": {"url": "data:image/tiff;base64,AAAA"}},
            {"type": "image_url", "image_url": {"url": "ftp://x/y.png"}},
            {"type": "video", "video": {}},
        ):
            with self.subTest(part=part["type"]), self.assertRaises(ProviderHTTPError) as ctx:
                _body(_req(messages=[{"role": "user", "content": [part]}]))
            self.assertEqual(ctx.exception.status_code, 400)

    def test_tool_round_trip(self):
        body = _body(_req(messages=[
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"x"}'}},
                {"id": "c2", "type": "function", "function": {"name": "get_time", "arguments": ""}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
            {"role": "tool", "tool_call_id": "c2", "content": [{"type": "text", "text": "noon"}]},
            {"role": "user", "content": "thanks"},
        ]))
        self.assertEqual(body["messages"][1], {"role": "assistant", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"x"}'}},
            {"id": "c2", "type": "function", "function": {"name": "get_time", "arguments": "{}"}},
        ]})
        self.assertEqual(body["messages"][2], {"role": "tool", "tool_results": [
            {"id": "c1", "type": "function", "content": "sunny", "name": "get_weather"},
            {"id": "c2", "type": "function", "content": "noon", "name": "get_time"},
        ]})
        self.assertEqual(body["messages"][3], {"role": "user", "contents": [_text("thanks")]})

    def test_assistant_text_with_tool_calls(self):
        body = _body(_req(messages=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "Checking.", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]))
        self.assertEqual(body["messages"][1]["contents"], [_text("Checking.")])

    def test_tool_image_becomes_follow_up_user_turn(self):
        body = _body(_req(messages=[
            {"role": "user", "content": "shot"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                                                  "function": {"name": "snap", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": [
                {"type": "text", "text": "ok"}, {"type": "image_url", "image_url": {"url": PNG_URI}}]},
        ]))
        self.assertEqual(body["messages"][2]["tool_results"][0]["content"], "ok")
        self.assertEqual(body["messages"][3]["contents"][0], _text("Image returned by tool call c1:"))
        self.assertEqual(body["messages"][3]["contents"][1]["image"]["type"], "base64")

    def test_empty_assistant_turn_skipped_and_errors(self):
        body = _body(_req(messages=[
            {"role": "user", "content": "a"}, {"role": "assistant", "content": ""}, {"role": "user", "content": "b"},
        ]))
        self.assertEqual(len(body["messages"]), 2)
        with self.assertRaises(ProviderHTTPError):
            _body(_req(messages=[{"role": "system", "content": "x"}]))
        with self.assertRaises(ProviderHTTPError):
            _body(_req(messages=[{"role": "tool", "content": "x"}]))

    def test_message_limit(self):
        _body(_req(messages=[{"role": "user", "content": "x"}] * 100))
        with self.assertRaises(ProviderHTTPError) as ctx:
            _body(_req(messages=[{"role": "user", "content": "x"}] * 101))
        self.assertIn("at most 100 messages", ctx.exception.message)


class ParameterTests(unittest.TestCase):
    def test_sampling_params_forwarded(self):
        # claude-sonnet-4-5 predates the 4.7 sampling-param removal, so nothing is scrubbed.
        body = _body(_req(temperature=0.2, top_p=0.9, stop=["x"], seed=1, user="u", top_k=7), model="claude-sonnet-4-5")
        self.assertEqual((body["temperature"], body["top_p"], body["seed"], body["top_k"]), (0.2, 0.9, 1, 7))
        self.assertEqual(body["stop_sequences"], ["x"])
        self.assertNotIn("user", body)
        self.assertEqual(_body(_req(stop="END"))["stop_sequences"], ["END"])
        self.assertEqual(_body(_req(extra_body={"top_k": 3}), model="claude-sonnet-4-5")["top_k"], 3)
        with self.assertRaises(ProviderHTTPError):
            _body(_req(stop=["a", "b", "c", "d", "e"]))

    def test_model_capability_rules_apply(self):
        req = _req(temperature=0.2, top_p=0.9, top_k=7, stop=["x"], seed=1)
        cases = {
            "claude-haiku-5-5": {"temperature", "top_p", "top_k"},
            "claude-opus-4-7": {"temperature", "top_p", "top_k"},
            "gpt-5.5": {"temperature", "top_p", "stop_sequences"},
            "gpt-4o": set(),
        }
        for model, dropped in cases.items():
            with self.subTest(model=model):
                body = _body(req, model=model)
                sent = {"temperature", "top_p", "top_k", "stop_sequences"} & body.keys()
                self.assertEqual(sent, {"temperature", "top_p", "top_k", "stop_sequences"} - dropped)
                self.assertEqual(body["seed"], 1)

    def test_unsupported_params_left_out(self):
        body = _body(_req(temperature=0.2, top_p=0.9, seed=1), unsupported={"temperature", "top_p"})
        self.assertNotIn("temperature", body)
        self.assertNotIn("top_p", body)
        self.assertEqual(body["seed"], 1)

    def test_n_becomes_variations(self):
        self.assertEqual(_body(_req(n=3))["variations"], 3)
        self.assertNotIn("variations", _body(_req(n=1)))
        for kwargs in ({"n": 6}, {"n": 2, "stream": True}):
            with self.subTest(**kwargs), self.assertRaises(ProviderHTTPError):
                _body(_req(n=kwargs["n"]), stream=kwargs.get("stream", False))
        with self.assertRaises(ProviderHTTPError):
            _body(_req(n=2), unsupported={"variations"})

    def test_max_tokens(self):
        self.assertNotIn("max_tokens", _body())
        self.assertEqual(_body(_req(max_tokens=100000), max_tokens_ceiling=4096)["max_tokens"], 4096)
        self.assertEqual(_body(_req(max_completion_tokens=50, max_tokens=99))["max_tokens"], 50)
        self.assertEqual(_body(max_tokens_ceiling=4096)["max_tokens"], 4096)
        with self.assertRaises(ProviderHTTPError):
            _body(_req(max_tokens=0))

    def test_reasoning(self):
        self.assertEqual(_body(_req(reasoning_effort="max"))["reasoning"], {"effort": "high"})
        self.assertEqual(_body(_req(reasoning_effort="minimal"))["reasoning"], {"effort": "minimal"})
        self.assertEqual(_body(_req(reasoning_effort="none"))["reasoning"], {"enabled": False})
        self.assertEqual(_body(_req(reasoning_effort="high"), effort_ceiling="low")["reasoning"], {"effort": "low"})
        self.assertNotIn("reasoning", _body(_req(reasoning_effort="high"), effort_ceiling="none"))
        self.assertNotIn("reasoning", _body(_req(reasoning_effort="high"), unsupported={"reasoning"}))

    def test_response_format_native(self):
        body = _body(_req(response_format={"type": "json_schema", "json_schema": {
            "name": "r", "strict": True, "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}}))
        self.assertEqual(body["response_format"], {"type": "json_schema", "json_schema": {
            "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}})
        self.assertNotIn("context", body)
        self.assertEqual(_body(_req(response_format={"type": "json_object"}))["response_format"], {"type": "json_object"})
        self.assertNotIn("response_format", _body(_req(response_format={"type": "text"})))

    def test_response_format_falls_back_to_instruction(self):
        body = _body(_req(response_format={"type": "json_schema",
                                           "json_schema": {"name": "r", "schema": {"type": "object"}}}),
                     unsupported={"response_format"})
        self.assertNotIn("response_format", body)
        self.assertIn('JSON Schema (named "r")', body["context"][-1]["text"])
        body = _body(_req(response_format={"type": "json_object"}), unsupported={"response_format"})
        self.assertIn("single valid JSON object", body["context"][-1]["text"])

    def test_web_search_options(self):
        body = _body(_req(web_search_options={
            "search_context_size": "low", "user_location": {"type": "approximate", "approximate": {
                "country": "IN", "city": "Chennai", "latitude": 13.08, "longitude": 80.27}}}))
        self.assertEqual(body["web_search_options"], {"search_context_size": "low", "user_location": {
            "country": "IN", "city": "Chennai", "latitude": 13, "longitude": 80}})


class ToolOptionTests(unittest.TestCase):
    TOOLS = [
        {"type": "function", "function": {"name": "a", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "b", "description": "B"}},
    ]

    def test_tools_sanitized(self):
        body = _body(_req(tools=self.TOOLS))
        self.assertEqual(body["tools"][1], {"type": "function", "function": {
            "name": "b", "description": "B", "parameters": {"type": "object", "properties": {}}}})
        self.assertNotIn("tool_choice", body)

    def test_tool_choice_mapping(self):
        self.assertEqual(_body(_req(tools=self.TOOLS, tool_choice="none"))["tool_choice"],
                         {"function": {"type": "none"}})
        self.assertEqual(_body(_req(tools=self.TOOLS, tool_choice="auto"))["tool_choice"],
                         {"function": {"type": "auto"}})
        self.assertEqual(_body(_req(tools=self.TOOLS, tool_choice="required"))["tool_choice"],
                         {"function": {"type": "any", "names": ["a", "b"]}})
        named = _body(_req(tools=self.TOOLS, tool_choice={"type": "function", "function": {"name": "b"}}))
        self.assertEqual(named["tool_choice"], {"function": {"type": "any", "names": ["b"]}})
        self.assertNotIn("tool_choice", _body(_req(tool_choice="auto")))

    def test_tool_choice_errors(self):
        with self.assertRaises(ProviderHTTPError):
            _body(_req(tools=self.TOOLS, tool_choice={"type": "function", "function": {"name": "zzz"}}))
        with self.assertRaises(ProviderHTTPError):
            _body(_req(tool_choice="required"))

    def test_tool_limit(self):
        tools = [{"type": "function", "function": {"name": f"t{i}"}} for i in range(129)]
        with self.assertRaises(ProviderHTTPError):
            _body(_req(tools=tools))


class EffortTests(unittest.TestCase):
    def test_effort_mapping(self):
        self.assertEqual(map_effort("none"), "none")
        self.assertIsNone(map_effort(None))
        self.assertEqual(map_effort("minimal"), "minimal")
        self.assertEqual(map_effort("xhigh"), "high")
        self.assertEqual(map_effort("high", ceiling="medium"), "medium")
        self.assertEqual(map_effort("low", ceiling="high"), "low")
        self.assertIsNone(map_effort("high", ceiling="none"))
        self.assertEqual(step_down_effort("high"), "medium")
        self.assertIsNone(step_down_effort("minimal"))
        with self.assertRaises(ProviderHTTPError):
            map_effort("extreme")


class SchemaSanitizerTests(unittest.TestCase):
    def test_rewrites(self):
        schema = {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "properties": {
                "n": {"type": "number", "minimum": 0.01, "maximum": 9.5, "exclusiveMaximum": True},
                "s": {"type": ["string", "null"], "deprecated": True, "x-ui": 1},
                "o": {"type": "string", "nullable": True},
                "t": {"type": "array", "items": [{"type": "string"}], "additionalItems": False},
                "big": {"type": "integer", "maximum": 1e30},
                "any": True,
                "never": False,
                "": {"type": "string"},
            },
            "dependencies": {"a": ["b"], "c": {"type": "object"}},
            "default": {"x-keep": 1},
        }
        out = sanitize_tool_schema(schema)
        self.assertIs(out["additionalProperties"], True)
        props = out["properties"]
        self.assertEqual(props["n"], {"type": "number", "minimum": 0, "exclusiveMaximum": 10})
        self.assertEqual(props["s"], {"anyOf": [{"type": "string"}, {"type": "null"}]})
        self.assertEqual(props["o"], {"anyOf": [{"type": "string"}, {"type": "null"}]})
        self.assertEqual(props["t"], {"type": "array", "prefixItems": [{"type": "string"}], "items": {"not": {}}})
        self.assertEqual(props["big"], {"type": "integer"})
        self.assertEqual(props["any"], {})
        self.assertEqual(props["never"], {"not": {}})
        self.assertNotIn("", props)
        self.assertEqual(out["dependentRequired"], {"a": ["b"]})
        self.assertEqual(out["default"], {"x-keep": 1})

    def test_root_forced_to_object(self):
        self.assertEqual(sanitize_tool_schema(None), {"type": "object", "properties": {}})
        self.assertEqual(sanitize_tool_schema({"properties": {}}), {"type": "object", "properties": {}})
        self.assertEqual(sanitize_tool_schema(["x"]), {"type": "object", "properties": {}})


class ResponseHelperTests(unittest.TestCase):
    def test_finish_reason(self):
        self.assertEqual(normalize_finish_reason("maxtokens", False), "length")
        self.assertEqual(normalize_finish_reason("MAX_TOKENS", False), "length")
        self.assertEqual(normalize_finish_reason("tool_calls", False), "tool_calls")
        self.assertEqual(normalize_finish_reason("end_turn", False), "stop")
        self.assertEqual(normalize_finish_reason(None, False), "stop")
        self.assertEqual(normalize_finish_reason("stop", True), "tool_calls")

    def test_usage(self):
        # Real gateway usage objects: prompt_tokens excludes cache reads and writes.
        write = build_usage({"reasoning_token": 8, "cache_write_tokens": 12614, "cache_write_1h_tokens": 0,
                             "completion_tokens": 8, "prompt_tokens": 31, "total_tokens": 39,
                             "cache_write_5m_tokens": 12614, "cached_tokens": 0})
        self.assertEqual((write.prompt_tokens, write.completion_tokens, write.total_tokens), (12645, 8, 12653))
        self.assertEqual(write.prompt_tokens_details, {"cached_tokens": 0})
        self.assertEqual(write.completion_tokens_details, {"reasoning_tokens": 8})
        self.assertEqual(write.cache_creation_input_tokens, 12614)
        read = build_usage({"completion_tokens": 8, "prompt_tokens": 31, "total_tokens": 39, "cached_tokens": 12614})
        self.assertEqual((read.prompt_tokens, read.total_tokens), (12645, 12653))
        self.assertEqual(read.prompt_tokens_details, {"cached_tokens": 12614})
        self.assertEqual(read.cache_read_input_tokens, 12614)
        self.assertEqual(build_usage({"prompt_tokens": 3, "completion_tokens": 4}).total_tokens, 7)
        self.assertEqual(build_usage(None).total_tokens, 0)

    def test_merge_usage_keeps_largest(self):
        merged = merge_usage({"prompt_tokens": 16, "completion_tokens": 9}, {"prompt_tokens": 16, "completion_tokens": 2})
        self.assertEqual(merged, {"prompt_tokens": 16, "completion_tokens": 9})
        self.assertEqual(merge_usage(None, {"a": 1}), {"a": 1})


if __name__ == "__main__":
    unittest.main()
