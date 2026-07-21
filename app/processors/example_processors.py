"""
Example processors demonstrating how to create custom request and response processors.

These examples show different patterns and use cases for the transformation system.
"""

import re
import logging
from datetime import datetime, timezone
from typing import Dict, Any
from app.transformation import RequestProcessor, ResponseProcessor
from app.openai_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionResponse,
    TextContent,
)

logger = logging.getLogger(__name__)


class CustomHeaderProcessor(RequestProcessor):
    """
    Example processor that adds custom headers to the transformation context.
    
    Configuration example:
    {
        "headers": {
            "X-Custom-Source": "llm-proxy",
            "X-Processing-Version": "1.0"
        }
    }
    """
    
    def is_applicable(self, data: Any) -> bool:
        """Apply to all request types."""
        return True
    
    async def process(self, request: Any, context: Dict[str, Any]) -> Any:
        """Add custom headers to the context."""
        custom_headers = self.processor_config.get("headers", {})
        
        if "custom_headers" not in context:
            context["custom_headers"] = {}
        
        context["custom_headers"].update(custom_headers)
        
        # Log the processing (optional)
        if custom_headers:
            print(f"CustomHeaderProcessor: Added headers {list(custom_headers.keys())}")
        
        return request


class TokenCountProcessor(RequestProcessor):
    """
    Example processor that estimates and logs token counts for requests.
    
    Configuration example:
    {
        "log_counts": true,
        "add_to_context": true
    }
    """
    
    def is_applicable(self, data: Any) -> bool:
        """Apply to chat completion requests."""
        return isinstance(data, ChatCompletionRequest)
    
    async def process(self, request: ChatCompletionRequest, context: Dict[str, Any]) -> ChatCompletionRequest:
        """Estimate token count for the request."""
        total_chars = 0
        message_count = len(request.messages)
        
        for message in request.messages:
            if isinstance(message.content, str):
                total_chars += len(message.content)
        
        # Rough token estimation (4 chars per token average)
        estimated_tokens = total_chars // 4
        
        # Add to context if configured
        if self.processor_config.get("add_to_context", False):
            context["estimated_input_tokens"] = estimated_tokens
            context["message_count"] = message_count
        
        # Log if configured
        if self.processor_config.get("log_counts", False):
            print(f"TokenCountProcessor: ~{estimated_tokens} tokens, {message_count} messages")
        
        return request


class ResponseTimestampProcessor(ResponseProcessor):
    """
    Example processor that adds processing timestamps to responses.
    
    Configuration example:
    {
        "add_processing_time": true,
        "timezone": "UTC"
    }
    """
    
    def is_applicable(self, data: Any) -> bool:
        """Apply to chat and completion responses."""
        return isinstance(data, (ChatCompletionResponse, CompletionResponse))
    
    async def process(self, response: Any, context: Dict[str, Any]) -> Any:
        """Add timestamp information to the response."""
        if self.processor_config.get("add_processing_time", True):
            # Add processing timestamp (timezone-aware; datetime.utcnow() is
            # deprecated in Python 3.12).
            processing_time = datetime.now(timezone.utc).isoformat()

            # Response models use extra="allow", so attach the metadata directly
            # rather than gating on a hasattr check that is always False.
            metadata = getattr(response, "metadata", None) or {}
            metadata["processed_at"] = processing_time
            metadata["processor"] = "ResponseTimestampProcessor"
            try:
                response.metadata = metadata
            except (ValueError, AttributeError):
                logger.debug("ResponseTimestampProcessor: could not set metadata on %s", type(response).__name__)

            # Also add to context for other processors
            context["processing_timestamp"] = processing_time

        return response


class ContentSanitizerProcessor(RequestProcessor):
    """
    Example processor that sanitizes content based on configurable rules.
    
    Configuration example:
    {
        "rules": [
            {
                "type": "remove_emails",
                "enabled": true
            },
            {
                "type": "replace_pattern",
                "pattern": "\\b\\d{4}-\\d{4}-\\d{4}-\\d{4}\\b",
                "replacement": "[CARD-NUMBER]",
                "enabled": true
            },
            {
                "type": "remove_urls",
                "enabled": false
            }
        ]
    }
    """
    
    def is_applicable(self, data: Any) -> bool:
        """Apply to chat completion requests."""
        return isinstance(data, ChatCompletionRequest)
    
    def _sanitize_text(self, text: str, rules: list) -> str:
        """Apply all enabled rules to a single text string.

        A malformed rule pattern is logged and skipped instead of aborting the
        whole request (re.error would otherwise bubble up).
        """
        for rule in rules:
            if not rule.get("enabled", True):
                continue

            rule_type = rule.get("type")

            try:
                if rule_type == "remove_emails":
                    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                    text = re.sub(email_pattern, "[EMAIL]", text)

                elif rule_type == "replace_pattern":
                    pattern = rule.get("pattern")
                    replacement = rule.get("replacement", "[REDACTED]")
                    if pattern:
                        text = re.sub(pattern, replacement, text)

                elif rule_type == "remove_urls":
                    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                    text = re.sub(url_pattern, "[URL]", text)
            except re.error as exc:
                logger.warning(
                    "ContentSanitizerProcessor: invalid pattern for rule %r, skipping: %s",
                    rule_type, exc,
                )

        return text

    async def process(self, request: ChatCompletionRequest, context: Dict[str, Any]) -> ChatCompletionRequest:
        """Sanitize content based on configured rules."""
        rules = self.processor_config.get("rules", [])
        sanitized_count = 0

        for message in request.messages:
            content = message.content
            if isinstance(content, str):
                new_content = self._sanitize_text(content, rules)
                if new_content != content:
                    message.content = new_content
                    sanitized_count += 1
            elif isinstance(content, list):
                # Multimodal message — sanitize each text part in place.
                changed = False
                for part in content:
                    part_text = part.text if isinstance(part, TextContent) else (
                        part.get("text") if isinstance(part, dict) and part.get("type") == "text" else None
                    )
                    if not isinstance(part_text, str):
                        continue
                    new_text = self._sanitize_text(part_text, rules)
                    if new_text != part_text:
                        if isinstance(part, TextContent):
                            part.text = new_text
                        else:
                            part["text"] = new_text
                        changed = True
                if changed:
                    sanitized_count += 1

        if sanitized_count > 0:
            context["sanitized_messages"] = sanitized_count
            logger.info("ContentSanitizerProcessor: Sanitized %d messages", sanitized_count)

        return request


class ModelOverrideProcessor(RequestProcessor):
    """
    Example processor that overrides model selection based on content analysis.
    
    Configuration example:
    {
        "rules": [
            {
                "condition": "long_context",
                "min_tokens": 8000,
                "target_model": "gpt-4-32k"
            },
            {
                "condition": "code_request",
                "keywords": ["code", "programming", "function", "class"],
                "target_model": "gpt-4"
            }
        ],
        "default_model": null
    }
    """
    
    def is_applicable(self, data: Any) -> bool:
        """Apply to requests with model field."""
        return hasattr(data, 'model')
    
    async def process(self, request: Any, context: Dict[str, Any]) -> Any:
        """Override model based on content analysis."""
        rules = self.processor_config.get("rules", [])
        original_model = request.model
        
        # Analyze content if it's a chat request
        if isinstance(request, ChatCompletionRequest):
            total_content = " ".join([
                msg.content for msg in request.messages 
                if isinstance(msg.content, str)
            ])
            
            estimated_tokens = len(total_content) // 4
            
            for rule in rules:
                condition = rule.get("condition")
                target_model = rule.get("target_model")
                
                if not target_model:
                    continue
                
                if condition == "long_context":
                    min_tokens = rule.get("min_tokens", 8000)
                    if estimated_tokens >= min_tokens:
                        request.model = target_model
                        context["model_override_reason"] = f"Long context ({estimated_tokens} tokens)"
                        break
                
                elif condition == "code_request":
                    keywords = rule.get("keywords", [])
                    content_lower = total_content.lower()
                    if any(keyword.lower() in content_lower for keyword in keywords):
                        request.model = target_model
                        context["model_override_reason"] = "Code-related request detected"
                        break
        
        # Log model change if it occurred
        if hasattr(request, 'model') and request.model != original_model:
            print(f"ModelOverrideProcessor: Changed model from {original_model} to {request.model}")
            context["original_model"] = original_model
        
        return request
