"""Local input-token estimate for Anthropic Messages payloads.

Used by ``/v1/messages/count_tokens`` when the upstream can't count, and for
the ``message_start`` usage of adapter streams. Text is tokenized with
tiktoken (``o200k_base``) when available; images and PDFs are charged a
fixed/size-based cost instead of tokenizing their base64 data. Any tiktoken
failure (not installed, offline encoding download, ...) falls back to the
~4 chars/token heuristic.

Set ``TIKTOKEN_CACHE_DIR`` to a pre-seeded directory for offline deployments.
"""

import json
import logging
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Roughly what a ~1MP image costs (OpenAI high-detail 1024x1024: 85 + 4*170).
IMAGE_TOKENS = 765
# PDFs: ~1,500 tokens per page and very roughly ~60KB per page of PDF.
PDF_TOKENS_PER_BYTE = 1500 / 60_000
MIN_DOCUMENT_TOKENS = 1500
MESSAGE_OVERHEAD_TOKENS = 3
REPLY_PRIMING_TOKENS = 3

_encoder_lock = threading.Lock()
_encoder: Optional[Callable[[str], int]] = None
_encoder_failed = False


def _heuristic_count(text: str) -> int:
    return (len(text) + 3) // 4 if text else 0


def _get_counter() -> Callable[[str], int]:
    global _encoder, _encoder_failed
    if _encoder is not None:
        return _encoder
    if _encoder_failed:
        return _heuristic_count
    with _encoder_lock:
        if _encoder is None and not _encoder_failed:
            try:
                import tiktoken

                encoding = tiktoken.get_encoding("o200k_base")

                def _count(text: str) -> int:
                    return len(encoding.encode(text, disallowed_special=())) if text else 0

                _count("warmup")
                _encoder = _count
            except Exception as e:  # ImportError, network errors, ...
                logger.info("tiktoken unavailable for token estimates (%s); using ~4 chars/token", e)
                _encoder_failed = True
    return _encoder or _heuristic_count


def count_text_tokens(text: str) -> int:
    try:
        return _get_counter()(text)
    except Exception:
        return _heuristic_count(text)


def _as_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return value


def _base64_bytes(data: Any) -> int:
    return (len(data) * 3) // 4 if isinstance(data, str) else 0


def _block_tokens(block: Any) -> int:
    block = _as_dict(block)
    if isinstance(block, str):
        return count_text_tokens(block)
    if not isinstance(block, dict):
        return count_text_tokens(str(block))

    block_type = block.get("type")
    if block_type == "text":
        return count_text_tokens(block.get("text", ""))
    if block_type == "image":
        return IMAGE_TOKENS
    if block_type == "document":
        source = block.get("source") or {}
        source_type = source.get("type")
        if source_type == "text":
            return count_text_tokens(source.get("data", ""))
        if source_type == "content":
            return _content_tokens(source.get("content"))
        if source_type == "base64":
            media_type = source.get("media_type") or "application/pdf"
            if media_type.startswith("image/"):
                return IMAGE_TOKENS
            return max(MIN_DOCUMENT_TOKENS, int(_base64_bytes(source.get("data")) * PDF_TOKENS_PER_BYTE))
        return MIN_DOCUMENT_TOKENS
    if block_type == "tool_use" or block_type == "server_tool_use":
        return count_text_tokens(block.get("name", "")) + count_text_tokens(
            json.dumps(block.get("input") or {}, ensure_ascii=False)
        )
    if block_type == "tool_result":
        return _content_tokens(block.get("content"))
    if block_type == "thinking":
        return count_text_tokens(block.get("thinking", ""))
    if block_type == "redacted_thinking":
        return 0
    if block_type == "search_result":
        return count_text_tokens(block.get("title", "")) + _content_tokens(block.get("content"))
    if block_type == "web_search_tool_result":
        content = block.get("content")
        if isinstance(content, list):
            return sum(
                count_text_tokens(f"{r.get('title', '')} {r.get('url', '')}")
                + _base64_bytes(r.get("encrypted_content")) // 4
                for r in content if isinstance(r, dict)
            )
        return 0
    # Unknown block: count its JSON, minus obviously opaque base64 payloads.
    trimmed = {k: v for k, v in block.items() if k not in ("data", "encrypted_content", "signature")}
    return count_text_tokens(json.dumps(trimmed, ensure_ascii=False, sort_keys=True))


def _content_tokens(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return count_text_tokens(content)
    if isinstance(content, list):
        return sum(_block_tokens(block) for block in content)
    return _block_tokens(content)


def estimate_anthropic_input_tokens(payload: Any) -> int:
    """Estimate input tokens for an Anthropic Messages request (dict or model)."""
    payload = _as_dict(payload)
    if not isinstance(payload, dict):
        return 0

    total = REPLY_PRIMING_TOKENS
    system = payload.get("system")
    if system:
        total += MESSAGE_OVERHEAD_TOKENS + _content_tokens(system)

    for message in payload.get("messages") or []:
        message = _as_dict(message)
        if isinstance(message, dict):
            total += MESSAGE_OVERHEAD_TOKENS + _content_tokens(message.get("content"))

    tools = payload.get("tools")
    if tools:
        serializable = [_as_dict(tool) for tool in tools]
        total += count_text_tokens(json.dumps(serializable, ensure_ascii=False, sort_keys=True))

    tool_choice = payload.get("tool_choice")
    if tool_choice:
        total += count_text_tokens(json.dumps(_as_dict(tool_choice), ensure_ascii=False, sort_keys=True))

    return total

