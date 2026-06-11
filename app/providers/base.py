from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, AsyncGenerator, Optional
import json
from app.openai_models import (
    ChatCompletionRequest, 
    CompletionRequest, 
    ChatCompletionResponse, 
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageEditRequest,
    ImageVariationRequest,
    ImageResponse,
    AudioSpeechRequest,
    AudioTranscriptionRequest,
    AudioTranslationRequest,
    AudioTranscriptionResponse,
    AudioTranslationResponse,
    ModelInfo,
    ResponsesCreateRequest,
    ResponsesCompactRequest,
    ResponsesInputTokensRequest,
    ResponseObject,
    ResponseDeletedObject,
    ResponseInputTokensResult,
    CompactedResponseObject,
    ResponseItemList
)
from app.anthropic_models import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
)


@dataclass
class AnthropicRequestMetadata:
    """Metadata about how a provider will serve an Anthropic request."""

    mode: str = "unsupported"
    transport: Optional[str] = None
    dropped_fields: List[str] = field(default_factory=list)


class ProviderHTTPError(Exception):
    """Provider error that preserves HTTP status, body, and response headers."""

    def __init__(
        self,
        status_code: int,
        message: str,
        body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.body = body
        self.headers = headers or {}


class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_type = self.__class__.__name__.lower().replace('provider', '')
        self.instance_name = config.get('name', 'default')
        
        # For OpenAI-compatible providers, use provider_name if available
        # For specialized providers (azure, bedrock, google), use provider_type
        if 'provider_name' in config and config['provider_name']:
            self.full_provider_name = f"{config['provider_name']}:{self.instance_name}"
        else:
            self.full_provider_name = f"{self.provider_type}:{self.instance_name}"
    
    @abstractmethod
    async def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models from the provider."""
        pass
    
    @abstractmethod
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle chat completion request."""
        pass
    
    @abstractmethod
    async def completion(self, request: CompletionRequest) -> CompletionResponse:
        """Handle text completion request."""
        pass
    
    @abstractmethod
    async def chat_completion_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """Handle streaming chat completion request - yields SSE formatted strings."""
        pass
    
    @abstractmethod
    async def completion_stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        """Handle streaming text completion request - yields SSE formatted strings."""
        pass
    
    async def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Handle embeddings request. Default implementation raises NotImplementedError."""
        raise NotImplementedError(f"Embeddings not supported by {self.__class__.__name__}")
    
    async def image_generation(self, request: ImageGenerationRequest) -> ImageResponse:
        """Handle image generation request. Default implementation raises NotImplementedError."""
        raise NotImplementedError(f"Image generation not supported by {self.__class__.__name__}")
    
    async def image_edit(self, request: ImageEditRequest) -> ImageResponse:
        """Handle image edit request. Default implementation raises NotImplementedError."""
        raise NotImplementedError(f"Image editing not supported by {self.__class__.__name__}")
    
    async def image_variation(self, request: ImageVariationRequest) -> ImageResponse:
        """Handle image variation request. Default implementation raises NotImplementedError."""
        raise NotImplementedError(f"Image variations not supported by {self.__class__.__name__}")
    
    async def audio_speech(self, request: AudioSpeechRequest) -> bytes:
        """Handle text-to-speech request. Default implementation raises NotImplementedError."""
        raise NotImplementedError(f"Text-to-speech not supported by {self.__class__.__name__}")
    
    async def audio_transcription(self, request: AudioTranscriptionRequest) -> AudioTranscriptionResponse:
        """Handle audio transcription request. Default implementation raises NotImplementedError."""
        raise NotImplementedError(f"Audio transcription not supported by {self.__class__.__name__}")
    
    async def audio_translation(self, request: AudioTranslationRequest) -> AudioTranslationResponse:
        """Handle audio translation request. Default implementation raises NotImplementedError."""
        raise NotImplementedError(f"Audio translation not supported by {self.__class__.__name__}")
    
    # ==================== RESPONSES API ====================
    
    async def responses_create(self, request: ResponsesCreateRequest) -> ResponseObject:
        """Handle Responses API create request. Default raises NotImplementedError."""
        raise NotImplementedError(f"Responses API not supported by {self.__class__.__name__}")
    
    async def responses_create_stream(self, request: ResponsesCreateRequest) -> AsyncGenerator[str, None]:
        """Handle streaming Responses API create request. Default raises NotImplementedError."""
        raise NotImplementedError(f"Responses API streaming not supported by {self.__class__.__name__}")
        yield  # Make this an async generator
    
    async def responses_retrieve(self, response_id: str, **kwargs) -> ResponseObject:
        """Retrieve a response by ID. Default raises NotImplementedError."""
        raise NotImplementedError(f"Responses API retrieve not supported by {self.__class__.__name__}")
    
    async def responses_delete(self, response_id: str) -> ResponseDeletedObject:
        """Delete a response by ID. Default raises NotImplementedError."""
        raise NotImplementedError(f"Responses API delete not supported by {self.__class__.__name__}")
    
    async def responses_cancel(self, response_id: str) -> ResponseObject:
        """Cancel a background response. Default raises NotImplementedError."""
        raise NotImplementedError(f"Responses API cancel not supported by {self.__class__.__name__}")
    
    async def responses_list_input_items(self, response_id: str, **kwargs) -> ResponseItemList:
        """List input items for a response. Default raises NotImplementedError."""
        raise NotImplementedError(f"Responses API list input items not supported by {self.__class__.__name__}")
    
    async def responses_input_tokens(self, request: ResponsesInputTokensRequest) -> ResponseInputTokensResult:
        """Count input tokens for a Responses API request. Default raises NotImplementedError."""
        raise NotImplementedError(f"Responses API input tokens not supported by {self.__class__.__name__}")
    
    async def responses_compact(self, request: ResponsesCompactRequest) -> CompactedResponseObject:
        """Compact a conversation. Default raises NotImplementedError."""
        raise NotImplementedError(f"Responses API compact not supported by {self.__class__.__name__}")
    
    # ==================== ANTHROPIC MESSAGES API ====================
    
    def get_supported_apis(self) -> List[str]:
        """Return list of API formats this provider supports.
        Override in subclasses. Default is OpenAI only."""
        return ["openai"]

    def get_supported_apis_for_model(self, model_name: str) -> List[str]:
        """Return supported APIs for a specific model.

        Providers can override this for model-specific routing. By default this
        mirrors provider-wide capabilities, except Anthropic support is gated on
        ``get_anthropic_mode_for_model()``.
        """
        supported = list(self.get_supported_apis())
        if "anthropic" in supported and self.get_anthropic_mode_for_model(model_name) == "unsupported":
            supported = [api for api in supported if api != "anthropic"]
        return supported

    def supports_api_for_model(self, model_name: str, api_name: str) -> bool:
        """Return True when the named model supports the requested API surface."""
        return api_name in self.get_supported_apis_for_model(model_name)

    def get_supported_endpoints(self) -> List[str]:
        """Return list of API endpoint paths this provider supports.
        Override in subclasses. Default includes only basic chat/completion/embedding endpoints."""
        return ["/v1/chat/completions", "/v1/completions", "/v1/embeddings"]

    def get_anthropic_mode_for_model(self, model_name: str) -> str:
        """Return how Anthropic requests are served for the given model.

        Modes:
        - ``native``: true Anthropic endpoint/shape upstream
        - ``adapter``: request translated onto another API
        - ``unsupported``: model should not be advertised on /v1/messages

        Subclasses with true native Anthropic endpoints (e.g. Bedrock) should
        override and return ``"native"`` for their Claude models.
        """
        if "anthropic" in self.get_supported_apis() and self._is_chat_capable_model(model_name):
            return "adapter"
        return "unsupported"

    def get_anthropic_request_metadata(
        self,
        request: AnthropicMessagesRequest,
        anthropic_beta: str | None = None,
    ) -> AnthropicRequestMetadata:
        """Return Anthropic routing metadata for the given request."""
        mode = self.get_anthropic_mode_for_model(request.model)
        return AnthropicRequestMetadata(
            mode=mode,
            transport="messages" if mode == "native" else None,
        )

    async def anthropic_messages(self, request: AnthropicMessagesRequest, anthropic_beta: str | None = None) -> AnthropicMessagesResponse:
        """Handle Anthropic Messages API request (non-streaming). Default raises NotImplementedError."""
        raise NotImplementedError(f"Anthropic Messages API not supported by {self.__class__.__name__}")
    
    async def anthropic_messages_stream(self, request: AnthropicMessagesRequest, anthropic_beta: str | None = None) -> AsyncGenerator[str, None]:
        """Handle streaming Anthropic Messages API request - yields SSE formatted strings.
        Default raises NotImplementedError."""
        raise NotImplementedError(f"Anthropic Messages API streaming not supported by {self.__class__.__name__}")
        yield  # Make this an async generator

    async def anthropic_count_tokens(self, request: Any) -> Dict[str, Any]:
        """Count Anthropic input tokens when the upstream supports it."""
        raise NotImplementedError(f"Anthropic count_tokens not supported by {self.__class__.__name__}")
    
    def is_enabled(self) -> bool:
        """Check if provider is enabled."""
        return self.config.get('enabled', False)
    
    def get_model_id(self, model_name: str) -> str:
        """Extract model ID from provider-prefixed model name."""
        # Remove provider prefix (e.g., "ollama/llama2" -> "llama2")
        if '/' in model_name:
            return model_name.split('/', 1)[1]
        return model_name
    
    def create_model_info(self, model_id: str, owned_by: str = None) -> ModelInfo:
        """Create ModelInfo object with provider prefix."""
        return ModelInfo(
            id=f"{self.full_provider_name}/{model_id}",
            created=1677610602,  # Static timestamp
            owned_by=owned_by or self.full_provider_name,
            provider=self.full_provider_name
        )

    def _is_chat_capable_model(self, model_name: str) -> bool:
        """Best-effort heuristic to hide clearly non-chat models from /v1/messages."""
        model_id = self.get_model_id(model_name).lower()
        unsupported_markers = (
            "text-embedding",
            "embedding",
            "embeddings",
            "imagen",
            "image-generation",
            "dall-e",
            "tts",
            "whisper",
            "transcribe",
            "transcription",
            "translation",
            "moderation",
            "rerank",
            "speech",
        )
        return not any(marker in model_id for marker in unsupported_markers)
    
    def format_sse_data(self, data: Dict[str, Any]) -> str:
        """Format data as Server-Sent Event."""
        try:
            # Ensure data is serializable
            if isinstance(data, str):
                # If data is already a string, treat it as an error message
                data = {"error": {"message": data, "type": "server_error"}}
            elif not isinstance(data, dict):
                # Convert non-dict objects to dict
                try:
                    # Try to convert to dict if it has model_dump method (Pydantic models)
                    if hasattr(data, 'model_dump'):
                        data = data.model_dump()
                    elif hasattr(data, 'dict'):
                        data = data.dict()
                    else:
                        data = {"error": {"message": str(data), "type": "server_error"}}
                except Exception:
                    data = {"error": {"message": str(data), "type": "server_error"}}
            
            # Ensure JSON serialization works
            json_str = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
            return f"data: {json_str}\n\n"
        except Exception as e:
            # Fallback for any serialization errors
            error_data = {
                "error": {
                    "message": f"Serialization error: {str(e)}",
                    "type": "server_error"
                }
            }
            try:
                return f"data: {json.dumps(error_data)}\n\n"
            except Exception:
                # Ultimate fallback
                return "data: {\"error\": {\"message\": \"Unknown serialization error\", \"type\": \"server_error\"}}\n\n"
    
    def format_sse_done(self) -> str:
        """Format the final SSE done message."""
        return "data: [DONE]\n\n"
    
    def format_sse_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """Format data as a typed Server-Sent Event with event: and data: lines.
        Used by the Responses API which uses typed events (e.g., event: response.output_text.delta)."""
        try:
            if isinstance(data, str):
                json_str = data
            elif hasattr(data, 'model_dump'):
                json_str = json.dumps(data.model_dump(), ensure_ascii=False, separators=(',', ':'))
            elif isinstance(data, dict):
                json_str = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
            else:
                json_str = json.dumps(str(data))
            return f"event: {event_type}\ndata: {json_str}\n\n"
        except Exception as e:
            error_data = json.dumps({"error": {"message": f"Serialization error: {str(e)}", "type": "server_error"}})
            return f"event: error\ndata: {error_data}\n\n"
