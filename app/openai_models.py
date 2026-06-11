from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, ConfigDict


# Content types for complex message handling
class TextContent(BaseModel):
    type: str = "text"
    text: str


class ImageUrl(BaseModel):
    url: str
    detail: Optional[str] = "auto"


class ImageContent(BaseModel):
    type: str = "image_url"
    image_url: ImageUrl


class ToolContent(BaseModel):
    type: str = "tool"
    text: str


# Function and tool call models
class Function(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ResponseFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ResponseFunction
    index: Optional[int] = None
    
    class Config:
        # Enable JSON serialization for this model
        json_encoders = {}
        # Ensure the model can be properly serialized
        arbitrary_types_allowed = False


class Tool(BaseModel):
    type: str = "function"
    function: Function


# Message types
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Optional[Union[str, List[Union[TextContent, ImageContent, ToolContent]]]] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    reasoning_content: Optional[str] = None


class UserMessage(BaseModel):
    role: str = "user"
    content: Union[str, List[Union[TextContent, ImageContent]]]


class AssistantMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str = "assistant"
    content: Optional[Union[str, List[Union[TextContent, ImageContent]]]] = None
    tool_calls: Optional[List[ToolCall]] = None
    reasoning_content: Optional[str] = None


class ToolMessage(BaseModel):
    role: str = "tool"
    content: Union[str, List[Union[TextContent, ToolContent]]]
    tool_call_id: str


class ChatMessageDelta(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = True


class ResponseFormat(BaseModel):
    type: str = "text"  # "text" or "json_object"
    json_schema: Optional[Dict[str, Any]] = None


class LogitBias(BaseModel):
    """Logit bias for specific tokens."""
    pass


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Union[ChatMessage, UserMessage, AssistantMessage, ToolMessage]]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    reasoning_effort: Optional[str] = None
    
    # Advanced parameters
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    user: Optional[str] = None
    
    class Config:
        extra = "allow"  # Allow extra fields to be passed through - handles ALL future parameters


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    
    class Config:
        extra = "allow"  # Allow extra fields to be passed through


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    provider: str
    description: Optional[str] = None
    max_tokens: Optional[int] = None


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class ChatCompletionChoice(BaseModel):
    model_config = ConfigDict(extra="allow")
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionStreamChoice(BaseModel):
    model_config = ConfigDict(extra="allow")
    index: int
    delta: ChatMessageDelta
    finish_reason: Optional[str] = None


class CompletionChoice(BaseModel):
    model_config = ConfigDict(extra="allow")
    index: int
    text: str
    finish_reason: Optional[str] = None


class CompletionStreamChoice(BaseModel):
    model_config = ConfigDict(extra="allow")
    index: int
    text: str
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow", exclude_none=False)
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]
    usage: Optional[Usage] = None


class CompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Optional[Usage] = None


class CompletionStreamResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionStreamChoice]


class ErrorMessage(BaseModel):
    message: str
    type: Optional[str] = None
    code: Optional[str] = None


class Error(BaseModel):
    error: ErrorMessage


class ErrorResponse(BaseModel):
    error: Dict[str, Any]


# Additional models for compatibility with reference implementation
class ChatResponseMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    reasoning_content: Optional[str] = None


class Choice(BaseModel):
    model_config = ConfigDict(extra="allow")
    index: int
    message: ChatResponseMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChoiceDelta(BaseModel):
    model_config = ConfigDict(extra="allow")
    index: int
    delta: ChatResponseMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


class ChatStreamResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChoiceDelta]
    usage: Optional[Usage] = None


# Embedding models
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str], List[int], List[List[int]]]
    model: str
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None
    
    class Config:
        extra = "allow"


class EmbeddingData(BaseModel):
    model_config = ConfigDict(extra="allow")
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    model_config = ConfigDict(extra="allow")
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


# Image generation models
class ImageGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = "dall-e-2"
    n: Optional[int] = 1
    quality: Optional[str] = "standard"  # "standard" or "hd"
    response_format: Optional[str] = "url"  # "url" or "b64_json"
    size: Optional[str] = "1024x1024"  # "256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"
    style: Optional[str] = "vivid"  # "vivid" or "natural"
    user: Optional[str] = None
    
    class Config:
        extra = "allow"


class ImageEditRequest(BaseModel):
    image: str  # File upload or base64
    prompt: str
    mask: Optional[str] = None  # File upload or base64
    model: Optional[str] = "dall-e-2"
    n: Optional[int] = 1
    response_format: Optional[str] = "url"
    size: Optional[str] = "1024x1024"
    user: Optional[str] = None
    
    class Config:
        extra = "allow"


class ImageVariationRequest(BaseModel):
    image: str  # File upload or base64
    model: Optional[str] = "dall-e-2"
    n: Optional[int] = 1
    response_format: Optional[str] = "url"
    size: Optional[str] = "1024x1024"
    user: Optional[str] = None
    
    class Config:
        extra = "allow"


class ImageData(BaseModel):
    model_config = ConfigDict(extra="allow")
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    created: int
    data: List[ImageData]


# Audio API models
class AudioSpeechRequest(BaseModel):
    model: str
    input: str
    voice: str = "alloy"  # "alloy", "echo", "fable", "onyx", "nova", "shimmer"
    response_format: Optional[str] = "mp3"  # "mp3", "opus", "aac", "flac", "wav", "pcm"
    speed: Optional[float] = 1.0  # 0.25 to 4.0
    
    class Config:
        extra = "allow"


class AudioTranscriptionRequest(BaseModel):
    file: str  # File upload or base64
    model: str
    language: Optional[str] = None  # ISO-639-1 format
    prompt: Optional[str] = None
    response_format: Optional[str] = "json"  # "json", "text", "srt", "verbose_json", "vtt"
    temperature: Optional[float] = 0
    timestamp_granularities: Optional[List[str]] = None  # ["word", "segment"]
    
    class Config:
        extra = "allow"


class AudioTranslationRequest(BaseModel):
    file: str  # File upload or base64
    model: str
    prompt: Optional[str] = None
    response_format: Optional[str] = "json"  # "json", "text", "srt", "verbose_json", "vtt"
    temperature: Optional[float] = 0
    
    class Config:
        extra = "allow"


class AudioTranscriptionWord(BaseModel):
    word: str
    start: float
    end: float


class AudioTranscriptionSegment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: Optional[List[AudioTranscriptionWord]] = None


class AudioTranscriptionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    words: Optional[List[AudioTranscriptionWord]] = None
    segments: Optional[List[AudioTranscriptionSegment]] = None


class AudioTranslationResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: Optional[List[AudioTranscriptionSegment]] = None


# Request type alias for compatibility
ChatRequest = ChatCompletionRequest


# ==================== RESPONSES API MODELS ====================

class ResponsesCreateRequest(BaseModel):
    """Request model for POST /v1/responses - Create a model response."""
    model: str
    input: Optional[Union[str, List[Any]]] = None
    instructions: Optional[str] = None
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    max_tool_calls: Optional[int] = None
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    text: Optional[Dict[str, Any]] = None
    reasoning: Optional[Dict[str, Any]] = None
    previous_response_id: Optional[str] = None
    conversation: Optional[Union[str, Dict[str, Any]]] = None
    store: Optional[bool] = None
    metadata: Optional[Dict[str, str]] = None
    background: Optional[bool] = None
    truncation: Optional[str] = None
    include: Optional[List[str]] = None
    service_tier: Optional[str] = None
    prompt: Optional[Dict[str, Any]] = None
    prompt_cache_key: Optional[str] = None
    prompt_cache_retention: Optional[str] = None
    safety_identifier: Optional[str] = None
    top_logprobs: Optional[int] = None
    context_management: Optional[List[Dict[str, Any]]] = None
    user: Optional[str] = None

    class Config:
        extra = "allow"


class ResponsesCompactRequest(BaseModel):
    """Request model for POST /v1/responses/compact - Compact a conversation."""
    model: str
    input: Optional[Union[str, List[Any]]] = None
    instructions: Optional[str] = None
    previous_response_id: Optional[str] = None

    class Config:
        extra = "allow"


class ResponsesInputTokensRequest(BaseModel):
    """Request model for POST /v1/responses/input_tokens - Count input tokens."""
    model: str
    input: Optional[Union[str, List[Any]]] = None
    instructions: Optional[str] = None
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    previous_response_id: Optional[str] = None
    conversation: Optional[Union[str, Dict[str, Any]]] = None
    reasoning: Optional[Dict[str, Any]] = None
    text: Optional[Dict[str, Any]] = None
    truncation: Optional[str] = None

    class Config:
        extra = "allow"


class ResponseUsage(BaseModel):
    """Usage info returned in Responses API."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_details: Optional[Dict[str, Any]] = None
    output_tokens_details: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class ResponseError(BaseModel):
    """Error object within a Response."""
    code: Optional[str] = None
    message: Optional[str] = None

    class Config:
        extra = "allow"


class ResponseObject(BaseModel):
    """Full response object from POST/GET /v1/responses."""
    id: str
    object: str = "response"
    created_at: Optional[int] = None
    status: Optional[str] = None
    completed_at: Optional[int] = None
    error: Optional[ResponseError] = None
    incomplete_details: Optional[Dict[str, Any]] = None
    instructions: Optional[str] = None
    model: Optional[str] = None
    output: Optional[List[Any]] = None
    output_text: Optional[str] = None
    parallel_tool_calls: Optional[bool] = None
    previous_response_id: Optional[str] = None
    conversation: Optional[Dict[str, Any]] = None
    reasoning: Optional[Dict[str, Any]] = None
    store: Optional[bool] = None
    temperature: Optional[float] = None
    text: Optional[Dict[str, Any]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    tools: Optional[List[Any]] = None
    top_p: Optional[float] = None
    truncation: Optional[str] = None
    background: Optional[bool] = None
    max_output_tokens: Optional[int] = None
    max_tool_calls: Optional[int] = None
    prompt: Optional[Dict[str, Any]] = None
    prompt_cache_key: Optional[str] = None
    prompt_cache_retention: Optional[str] = None
    safety_identifier: Optional[str] = None
    service_tier: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None
    usage: Optional[ResponseUsage] = None
    top_logprobs: Optional[int] = None
    user: Optional[str] = None

    class Config:
        extra = "allow"


class ResponseDeletedObject(BaseModel):
    """Response from DELETE /v1/responses/{response_id}."""
    model_config = ConfigDict(extra="allow")
    id: str
    object: str = "response"
    deleted: bool = True


class ResponseInputTokensResult(BaseModel):
    """Response from POST /v1/responses/input_tokens."""
    model_config = ConfigDict(extra="allow")
    object: str = "response.input_tokens"
    input_tokens: int


class CompactedResponseObject(BaseModel):
    """Response from POST /v1/responses/compact."""
    id: str
    object: str = "response.compaction"
    created_at: Optional[int] = None
    output: Optional[List[Any]] = None
    usage: Optional[ResponseUsage] = None

    class Config:
        extra = "allow"


class ResponseItemList(BaseModel):
    """Response from GET /v1/responses/{response_id}/input_items."""
    object: str = "list"
    data: List[Any] = []
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False

    class Config:
        extra = "allow"
