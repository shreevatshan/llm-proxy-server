from abc import ABC, abstractmethod
from contextvars import ContextVar
import logging
from typing import List, Dict, Any, AsyncGenerator
from openai import AsyncOpenAI
from app.providers.base import BaseProvider

# When True, provider methods will NOT overwrite the upstream model name
# in responses.  Set by the Azure OpenAI routes so the native model
# version string (e.g. "gpt-4.1-2025-04-14") passes through untouched.
preserve_upstream_model: ContextVar[bool] = ContextVar(
    "preserve_upstream_model", default=True
)

# Extra HTTP headers to forward to the upstream provider (e.g. Azure "aoai-*"
# preview feature headers).  Set by middleware; read by _prepare_request_dict
# so that all SDK create() calls automatically include them.
extra_request_headers: ContextVar[dict] = ContextVar(
    "extra_request_headers", default={}
)
from app.openai_models import (
    ChatCompletionRequest,
    CompletionRequest,
    ChatCompletionResponse,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    AudioSpeechRequest,
    AudioTranscriptionRequest,
    AudioTranslationRequest,
    AudioTranscriptionResponse,
    AudioTranslationResponse,
    ImageGenerationRequest,
    ImageEditRequest,
    ImageVariationRequest,
    ImageResponse,
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

logger = logging.getLogger(__name__)

_DEFAULT_API_KEY = "not-required"


class OpenAICompatibleProvider(BaseProvider):
    """
    Base class for providers that use OpenAI SDK for API calls.
    Provides common OpenAI-compatible request/response handling for all methods.
    
    This class handles:
    - Automatic parameter conversion using model_dump()/dict()
    - Direct SDK response handling
    - Consistent streaming implementation
    - Unified error handling
    - All OpenAI API methods with identical behavior
    
    Subclasses only need to:
    - Initialize the OpenAI client (self.client)
    - Implement get_model_id() for model name transformation
    - Implement get_available_models() for provider-specific model discovery
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Subclasses must initialize self.client
        self.client: AsyncOpenAI = None
        # Optional separate client for Responses API (e.g., Azure uses a different base URL)
        self._responses_client: AsyncOpenAI = None
    
    def _get_responses_client(self) -> AsyncOpenAI:
        """Return the client to use for Responses API calls.
        
        By default returns self.client. Subclasses (e.g. AzureProvider) can set
        self._responses_client to use a different client for the Responses API
        which may require a different base URL or auth configuration.
        """
        return self._responses_client if self._responses_client is not None else self.client

    # ==================== Initialization helpers ====================

    def _init_openai_client(self) -> None:
        """Initialize AsyncOpenAI client when OpenAI API is supported."""
        if "openai" not in getattr(self, "_supported_apis", []):
            return
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=getattr(self, "api_key", None) or _DEFAULT_API_KEY,
        )
    
    @abstractmethod
    def get_model_id(self, model_name: str) -> str:
        """Transform provider-prefixed model name to internal model ID."""
        pass
    
    def _prepare_request_dict(self, request: Any, model_override: str = None) -> Dict[str, Any]:
        """
        Common method to prepare request dictionary from Pydantic model.
        Handles all parameters automatically using model_dump() or dict().
        """
        # Convert Pydantic request to dict - this handles ALL parameters automatically
        request_dict = request.model_dump() if hasattr(request, 'model_dump') else request.dict()
        
        # Override model if specified (for deployment mapping)
        if model_override:
            request_dict["model"] = model_override
        
        # Remove None values and empty lists/dicts for cleaner API calls
        request_dict = {
            k: v for k, v in request_dict.items() 
            if v is not None and v != [] and v != {}
        }
        
        # If max_tokens is set, replace it with max_completion_tokens for newer models
        # This ensures compatibility with models like GPT-4o, o1-preview that don't support max_tokens
        if "max_tokens" in request_dict and request_dict["max_tokens"] is not None:
            request_dict["max_completion_tokens"] = request_dict["max_tokens"]
            del request_dict["max_tokens"]  # Remove max_tokens to avoid conflicts
        
        custom_provider = getattr(self, 'custom_provider_name', '').lower()

        # LMStudio uses "structured" instead of OpenAI's "response_format" for structured output.
        # Translate response_format → structured when targeting lmstudio.
        if custom_provider.startswith("lmstudio") and "response_format" in request_dict:
            response_format = request_dict.pop("response_format")
            fmt_type = response_format.get("type") if isinstance(response_format, dict) else None
            if fmt_type == "json_schema":
                json_schema_wrapper = response_format.get("json_schema", {})
                # The schema is nested under "schema" in the OpenAI spec
                schema = json_schema_wrapper.get("schema", json_schema_wrapper)
                if schema:
                    # Use extra_body so the OpenAI SDK forwards this non-standard
                    # parameter in the HTTP request body without raising an error
                    if "extra_body" not in request_dict:
                        request_dict["extra_body"] = {}
                    request_dict["extra_body"]["structured"] = schema
            # json_object type → no "structured" key needed; LMStudio will respond with JSON
            # via system prompt or other means; drop the unsupported response_format key

        # Clean up messages to remove null fields for all providers.
        # Many backends (LlamaCpp, vLLM, custom servers) are strict about JSON
        # and reject null values for string-typed fields like tool_call_id.
        # Sending null for absent optional fields is never useful — omitting
        # the key entirely is the correct OpenAI-compatible behaviour.
        if "messages" in request_dict:
            cleaned_messages = []
            for msg in request_dict["messages"]:
                if isinstance(msg, dict):
                    # Remove None/null values from each message
                    cleaned_msg = {k: v for k, v in msg.items() if v is not None}
                    cleaned_messages.append(cleaned_msg)
                else:
                    cleaned_messages.append(msg)
            request_dict["messages"] = cleaned_messages
        
        # Inject extra headers from ContextVar (e.g. Azure aoai-* preview headers).
        # The OpenAI SDK accepts an extra_headers kwarg on all create() calls.
        _extra_hdrs = extra_request_headers.get({})
        if _extra_hdrs:
            request_dict["extra_headers"] = {
                **request_dict.get("extra_headers", {}),
                **_extra_hdrs,
            }
        
        return request_dict
    
    def _preserve_original_model_name(self, response: Any, original_model: str) -> Any:
        """
        Preserve the original model name in the response.
        Works with both direct response objects and dict responses.
        """
        if hasattr(response, 'model'):
            response.model = original_model
        elif isinstance(response, dict) and 'model' in response:
            response['model'] = original_model
        return response

    # ==================== Model discovery helpers ====================

    async def _fetch_openai_models(self) -> List[ModelInfo] | None:
        """Fetch models via OpenAI-compatible /v1/models endpoint."""
        if "openai" not in getattr(self, "_supported_apis", []) or not getattr(self, "client", None):
            return None
        try:
            response = await self.client.models.list()
            provider_name = getattr(self, "custom_provider_name", None) or getattr(self, "provider_type", "provider")
            return [
                self.create_model_info(m.id, provider_name)
                for m in response.data if m.id
            ]
        except Exception as e:
            provider_name = getattr(self, "custom_provider_name", None) or getattr(self, "provider_type", "provider")
            logger.warning(f"Error fetching models via OpenAI API from {provider_name}: {e}")
            return None
    
    # ==================== CHAT COMPLETION ====================
    
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle chat completion request using OpenAI SDK."""
        model_id = self.get_model_id(request.model)
        request_dict = self._prepare_request_dict(request, model_id)
        
        try:
            # Remove any 'options' parameter that might have been passed through
            # Some providers like Ollama don't support this parameter
            request_dict.pop("options", None)
            
            response = await self.client.chat.completions.create(**request_dict)
            
            # Convert OpenAI SDK response to our Pydantic model to ensure proper serialization
            # Use exclude_unset=True to only include fields actually returned by upstream,
            # preserving response fidelity (e.g. annotations, refusal) and avoiding
            # extra null fields (e.g. reasoning_content, tool_call_id) not in the original.
            response_dict = response.model_dump(exclude_unset=True) if hasattr(response, 'model_dump') else response.dict()
            if not preserve_upstream_model.get(False):
                response_dict["model"] = request.model  # Preserve original model name
            
            return ChatCompletionResponse(**response_dict)
        except Exception as e:
            raise Exception(f"{self.provider_type} chat completion error: {str(e)}")
    
    async def chat_completion_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """Handle streaming chat completion request using OpenAI SDK."""
        model_id = self.get_model_id(request.model)
        request_dict = self._prepare_request_dict(request, model_id)
        request_dict["stream"] = True  # Ensure streaming is enabled
        
        # Ensure stream_options includes usage by default if not explicitly set
        if "stream_options" not in request_dict or request_dict["stream_options"] is None:
            request_dict["stream_options"] = {"include_usage": True}
        elif isinstance(request_dict["stream_options"], dict) and "include_usage" not in request_dict["stream_options"]:
            request_dict["stream_options"]["include_usage"] = True
        
        try:
            # Remove any 'options' parameter that might have been passed through
            # Some providers like Ollama don't support this parameter
            request_dict.pop("options", None)
            
            stream = await self.client.chat.completions.create(**request_dict)
            
            async for chunk in stream:
                # Skip if chunk is a string or None
                if isinstance(chunk, str) or chunk is None:
                    continue
                
                # Convert OpenAI chunk to dict for SSE serialization
                try:
                    chunk_dict = chunk.model_dump(exclude_unset=True) if hasattr(chunk, 'model_dump') else chunk.dict()
                    if not preserve_upstream_model.get(False):
                        chunk_dict["model"] = request.model  # Use original model name
                    
                    # Yield as SSE format
                    yield self.format_sse_data(chunk_dict)
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    error_data = {
                        "error": {
                            "message": f"Chunk processing error: {str(e)}",
                            "type": "server_error"
                        }
                    }
                    yield self.format_sse_data(error_data)
            
            # Send final done message
            yield self.format_sse_done()
            
        except Exception as e:
            error_data = {
                "error": {
                    "message": f"{self.provider_type} chat completion stream error: {str(e)}",
                    "type": "server_error"
                }
            }
            yield self.format_sse_data(error_data)
    
    # ==================== TEXT COMPLETION ====================
    
    async def completion(self, request: CompletionRequest) -> CompletionResponse:
        """Handle text completion request using OpenAI SDK."""
        model_id = self.get_model_id(request.model)
        request_dict = self._prepare_request_dict(request, model_id)
        
        try:
            # Remove any 'options' parameter that might have been passed through
            # Some providers like Ollama don't support this parameter
            request_dict.pop("options", None)
            
            response = await self.client.completions.create(**request_dict)
            
            # Convert OpenAI SDK response to our Pydantic model to ensure proper serialization
            response_dict = response.model_dump(exclude_unset=True) if hasattr(response, 'model_dump') else response.dict()
            if not preserve_upstream_model.get(False):
                response_dict["model"] = request.model  # Preserve original model name
            
            return CompletionResponse(**response_dict)
        except Exception as e:
            raise Exception(f"{self.provider_type} completion error: {str(e)}")
    
    async def completion_stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        """Handle streaming text completion request using OpenAI SDK."""
        model_id = self.get_model_id(request.model)
        request_dict = self._prepare_request_dict(request, model_id)
        request_dict["stream"] = True  # Ensure streaming is enabled
        
        # Ensure stream_options includes usage by default if not explicitly set
        if "stream_options" not in request_dict or request_dict["stream_options"] is None:
            request_dict["stream_options"] = {"include_usage": True}
        elif isinstance(request_dict["stream_options"], dict) and "include_usage" not in request_dict["stream_options"]:
            request_dict["stream_options"]["include_usage"] = True
        
        try:
            # Remove any 'options' parameter that might have been passed through
            # Some providers like Ollama don't support this parameter
            request_dict.pop("options", None)
            
            stream = await self.client.completions.create(**request_dict)
            
            async for chunk in stream:
                # Skip if chunk is a string or None
                if isinstance(chunk, str) or chunk is None:
                    continue
                
                # Convert OpenAI chunk to dict for SSE serialization
                try:
                    chunk_dict = chunk.model_dump(exclude_unset=True) if hasattr(chunk, 'model_dump') else chunk.dict()
                    if not preserve_upstream_model.get(False):
                        chunk_dict["model"] = request.model  # Use original model name
                    
                    # Yield as SSE format
                    yield self.format_sse_data(chunk_dict)
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    error_data = {
                        "error": {
                            "message": f"Chunk processing error: {str(e)}",
                            "type": "server_error"
                        }
                    }
                    yield self.format_sse_data(error_data)
            
            # Send final done message
            yield self.format_sse_done()
            
        except Exception as e:
            error_data = {
                "error": {
                    "message": f"{self.provider_type} completion stream error: {str(e)}",
                    "type": "server_error"
                }
            }
            yield self.format_sse_data(error_data)
    
    # ==================== EMBEDDINGS ====================
    
    async def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Handle embeddings request using OpenAI SDK."""
        model_id = self.get_model_id(request.model)
        request_dict = self._prepare_request_dict(request, model_id)
        
        try:
            # Remove any 'options' parameter that might have been passed through
            # Some providers like Ollama don't support this parameter
            request_dict.pop("options", None)
            
            response = await self.client.embeddings.create(**request_dict)
            
            # Convert response and preserve model name
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
            if not preserve_upstream_model.get(False):
                response_dict["model"] = request.model
            
            return EmbeddingResponse(**response_dict)
        except Exception as e:
            raise Exception(f"{self.provider_type} embeddings error: {str(e)}")
    
    # ==================== AUDIO METHODS ====================
    
    async def audio_speech(self, request: AudioSpeechRequest) -> bytes:
        """Handle text-to-speech request using OpenAI SDK."""
        model_id = self.get_model_id(request.model)
        request_dict = self._prepare_request_dict(request, model_id)
        
        try:
            response = await self.client.audio.speech.create(**request_dict)
            return response.content
        except Exception as e:
            raise Exception(f"{self.provider_type} audio speech error: {str(e)}")
    
    async def audio_transcription(self, request: AudioTranscriptionRequest) -> AudioTranscriptionResponse:
        """Handle audio transcription request using OpenAI SDK."""
        model_id = self.get_model_id(request.model)
        request_dict = self._prepare_request_dict(request, model_id)
        
        try:
            response = await self.client.audio.transcriptions.create(**request_dict)
            
            # Handle different response formats
            if request.response_format in ["srt", "vtt", "text"]:
                # For non-JSON formats, return as text
                return AudioTranscriptionResponse(text=str(response))
            else:
                # For JSON format (default), convert response to our model
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
                return AudioTranscriptionResponse(**response_dict)
        except Exception as e:
            raise Exception(f"{self.provider_type} audio transcription error: {str(e)}")
    
    async def audio_translation(self, request: AudioTranslationRequest) -> AudioTranslationResponse:
        """Handle audio translation request using OpenAI SDK."""
        model_id = self.get_model_id(request.model)
        request_dict = self._prepare_request_dict(request, model_id)
        
        try:
            response = await self.client.audio.translations.create(**request_dict)
            
            # Handle different response formats
            if request.response_format in ["srt", "vtt", "text"]:
                # For non-JSON formats, return as text
                return AudioTranslationResponse(text=str(response))
            else:
                # For JSON format (default), convert response to our model
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
                return AudioTranslationResponse(**response_dict)
        except Exception as e:
            raise Exception(f"{self.provider_type} audio translation error: {str(e)}")
    
    # ==================== IMAGE METHODS ====================
    
    async def image_generation(self, request: ImageGenerationRequest) -> ImageResponse:
        """Handle image generation request using OpenAI SDK."""
        model_id = self.get_model_id(request.model)
        request_dict = self._prepare_request_dict(request, model_id)
        
        try:
            response = await self.client.images.generate(**request_dict)
            
            # Convert OpenAI response to our format
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
            return ImageResponse(**response_dict)
        except Exception as e:
            raise Exception(f"{self.provider_type} image generation error: {str(e)}")
    
    async def image_edit(self, request: ImageEditRequest) -> ImageResponse:
        """Handle image edit request using OpenAI SDK."""
        model_id = self.get_model_id(request.model)
        request_dict = self._prepare_request_dict(request, model_id)
        
        try:
            response = await self.client.images.edit(**request_dict)
            
            # Convert OpenAI response to our format
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
            return ImageResponse(**response_dict)
        except Exception as e:
            raise Exception(f"{self.provider_type} image edit error: {str(e)}")
    
    async def image_variation(self, request: ImageVariationRequest) -> ImageResponse:
        """Handle image variation request using OpenAI SDK."""
        model_id = self.get_model_id(request.model)
        request_dict = self._prepare_request_dict(request, model_id)
        
        try:
            response = await self.client.images.create_variation(**request_dict)
            
            # Convert OpenAI response to our format
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
            return ImageResponse(**response_dict)
        except Exception as e:
            raise Exception(f"{self.provider_type} image variation error: {str(e)}")
    
    # ==================== RESPONSES API ====================
    
    async def responses_create(self, request: ResponsesCreateRequest) -> ResponseObject:
        """Handle Responses API create request using OpenAI SDK."""
        model_id = self.get_model_id(request.model)
        request_dict = self._prepare_request_dict(request, model_id)
        
        # Remove stream flag for non-streaming path
        request_dict.pop("stream", None)
        request_dict.pop("stream_options", None)
        
        try:
            response = await self._get_responses_client().responses.create(**request_dict)
            
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
            if not preserve_upstream_model.get(False):
                response_dict["model"] = request.model  # Preserve original model name
            
            return ResponseObject(**response_dict)
        except Exception as e:
            raise Exception(f"{self.provider_type} responses create error: {str(e)}")
    
    async def responses_create_stream(self, request: ResponsesCreateRequest) -> AsyncGenerator[str, None]:
        """Handle streaming Responses API create request using OpenAI SDK."""
        model_id = self.get_model_id(request.model)
        request_dict = self._prepare_request_dict(request, model_id)
        request_dict["stream"] = True  # Ensure streaming is enabled
        
        try:
            stream = await self._get_responses_client().responses.create(**request_dict)
            
            async for event in stream:
                try:
                    event_type = getattr(event, 'type', 'unknown')
                    event_dict = event.model_dump() if hasattr(event, 'model_dump') else event.dict()
                    
                    # Preserve original model name in response.created events
                    if not preserve_upstream_model.get(False):
                        if event_type == 'response.created' and 'response' in event_dict:
                            event_dict['response']['model'] = request.model
                    
                    yield self.format_sse_event(event_type, event_dict)
                except Exception as e:
                    print(f"Error processing responses stream event: {e}")
                    error_data = {
                        "error": {
                            "message": f"Event processing error: {str(e)}",
                            "type": "server_error"
                        }
                    }
                    yield self.format_sse_event("error", error_data)
            
            # Responses API streaming does NOT use data: [DONE]
            # The stream ends with response.completed / response.failed / response.incomplete
            
        except Exception as e:
            error_data = {
                "error": {
                    "message": f"{self.provider_type} responses stream error: {str(e)}",
                    "type": "server_error"
                }
            }
            yield self.format_sse_event("error", error_data)
    
    async def responses_retrieve(self, response_id: str, **kwargs) -> ResponseObject:
        """Retrieve a response by ID using OpenAI SDK."""
        try:
            response = await self._get_responses_client().responses.retrieve(response_id, **kwargs)
            
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
            return ResponseObject(**response_dict)
        except Exception as e:
            raise Exception(f"{self.provider_type} responses retrieve error: {str(e)}")
    
    async def responses_delete(self, response_id: str) -> ResponseDeletedObject:
        """Delete a response by ID using OpenAI SDK."""
        try:
            response = await self._get_responses_client().responses.delete(response_id)
            
            if hasattr(response, 'model_dump'):
                response_dict = response.model_dump()
            elif hasattr(response, 'dict'):
                response_dict = response.dict()
            elif isinstance(response, dict):
                response_dict = response
            else:
                response_dict = {"id": response_id, "object": "response", "deleted": True}
            
            return ResponseDeletedObject(**response_dict)
        except Exception as e:
            raise Exception(f"{self.provider_type} responses delete error: {str(e)}")
    
    async def responses_cancel(self, response_id: str) -> ResponseObject:
        """Cancel a background response using OpenAI SDK."""
        try:
            response = await self._get_responses_client().responses.cancel(response_id)
            
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
            return ResponseObject(**response_dict)
        except Exception as e:
            raise Exception(f"{self.provider_type} responses cancel error: {str(e)}")
    
    async def responses_list_input_items(self, response_id: str, **kwargs) -> ResponseItemList:
        """List input items for a response using OpenAI SDK."""
        try:
            response = await self._get_responses_client().responses.input_items.list(response_id, **kwargs)
            
            if hasattr(response, 'model_dump'):
                response_dict = response.model_dump()
            elif hasattr(response, 'dict'):
                response_dict = response.dict()
            elif isinstance(response, dict):
                response_dict = response
            else:
                # Handle paginated response object
                items = []
                if hasattr(response, 'data'):
                    items = [item.model_dump() if hasattr(item, 'model_dump') else item for item in response.data]
                response_dict = {
                    "object": "list",
                    "data": items,
                    "first_id": getattr(response, 'first_id', None),
                    "last_id": getattr(response, 'last_id', None),
                    "has_more": getattr(response, 'has_more', False)
                }
            
            return ResponseItemList(**response_dict)
        except Exception as e:
            raise Exception(f"{self.provider_type} responses list input items error: {str(e)}")
    
    async def responses_input_tokens(self, request: ResponsesInputTokensRequest) -> ResponseInputTokensResult:
        """Count input tokens for a Responses API request using OpenAI SDK."""
        model_id = self.get_model_id(request.model)
        request_dict = self._prepare_request_dict(request, model_id)
        
        try:
            response = await self._get_responses_client().responses.input_tokens.create(**request_dict)
            
            if hasattr(response, 'model_dump'):
                response_dict = response.model_dump()
            elif hasattr(response, 'dict'):
                response_dict = response.dict()
            elif isinstance(response, dict):
                response_dict = response
            else:
                response_dict = {
                    "object": "response.input_tokens",
                    "input_tokens": getattr(response, 'input_tokens', 0)
                }
            
            return ResponseInputTokensResult(**response_dict)
        except Exception as e:
            raise Exception(f"{self.provider_type} responses input tokens error: {str(e)}")
    
    async def responses_compact(self, request: ResponsesCompactRequest) -> CompactedResponseObject:
        """Compact a conversation using OpenAI SDK."""
        model_id = self.get_model_id(request.model)
        request_dict = self._prepare_request_dict(request, model_id)
        
        try:
            response = await self._get_responses_client().responses.compact.create(**request_dict)
            
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
            return CompactedResponseObject(**response_dict)
        except Exception as e:
            raise Exception(f"{self.provider_type} responses compact error: {str(e)}")
