"""Google Generative AI Provider for LLM Proxy Server."""

import os
from typing import Dict, List, Any, Optional, Union

from openai import AsyncOpenAI

from app.openai_models import ChatCompletionRequest, ModelInfo
from app.providers.anthropic_adapter import (
    anthropic_adapter_messages,
    anthropic_adapter_messages_stream,
    prepare_anthropic_adapter_request,
)
from app.providers.base import AnthropicRequestMetadata
from app.providers.openai_compatible import OpenAICompatibleProvider


class GoogleProvider(OpenAICompatibleProvider):
    """Google Generative AI provider using OpenAI SDK with Google's OpenAI-compatible API."""
    
    # Parameters not supported by Google's API
    UNSUPPORTED_PARAMS = {
        'frequency_penalty',
        'presence_penalty',
        'logit_bias',
        'logprobs',
        'top_logprobs',
        'suffix',
        'best_of',
        'echo',
        'user',
        'stream_options',
        'reasoning_effort',
        'parallel_tool_calls',
        'response_format',
    }
    
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Google provider with OpenAI-compatible configuration."""
        super().__init__(config)
        
        # Get API key from config or environment
        self.api_key = config.get('api_key') or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or provide in config.")
        
        # Get base URL from config or use default Google OpenAI-compatible endpoint
        self.base_url = config.get('base_url') or "https://generativelanguage.googleapis.com/v1beta/openai/"
        
        # Initialize OpenAI client with configurable endpoint
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        #print(f"[GOOGLE_PROVIDER] Initialized with OpenAI-compatible API at {self.base_url}")


    def _prepare_request_dict(self, request: Union[ChatCompletionRequest, Any], model_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare request dictionary for Google API, filtering out unsupported parameters.
        
        Google's OpenAI-compatible API doesn't support certain OpenAI parameters like
        frequency_penalty, presence_penalty, logit_bias, etc.
        """
        # Get the base request dict from parent class
        request_dict: Dict[str, Any] = super()._prepare_request_dict(request, model_override)

        if "max_completion_tokens" in request_dict and "max_tokens" not in request_dict:
            request_dict["max_tokens"] = request_dict.pop("max_completion_tokens")
        
        # Filter out unsupported parameters
        filtered_dict: Dict[str, Any] = {
            k: v for k, v in request_dict.items() 
            if k not in self.UNSUPPORTED_PARAMS
        }
        
        # Log if any parameters were filtered out (for debugging)
        removed_params: set = set(request_dict.keys()) - set(filtered_dict.keys())
        #if removed_params:
            #print(f"[GOOGLE_PROVIDER] Filtered out unsupported parameters: {removed_params}")
        
        return filtered_dict

    def get_model_id(self, model_name: str) -> str:
        """Get the model ID for Google models."""
        original_model: str = model_name
        
        # Google models can be used directly with their names
        # Remove any provider prefix if present (e.g., "primary:gemini-2.5-pro" -> "gemini-2.5-pro")
        if ':' in model_name:
            model_name = model_name.split(':', 1)[1]
        
        # Remove models/ prefix if present (e.g., "models/gemini-2.5-pro" -> "gemini-2.5-pro")
        if model_name.startswith('models/'):
            model_name = model_name[7:]
        
        # Remove any "primary/" or other prefixes before the actual model name
        if '/' in model_name:
            # Extract just the model name after the last slash
            model_name = model_name.split('/')[-1]
        
        # print(f"[GOOGLE_PROVIDER] Model name transformation: '{original_model}' -> '{model_name}'")
        return model_name

    def get_supported_endpoints(self) -> List[str]:
        """Google's OpenAI-compatible API supports chat completions, embeddings,
        and image generation but not responses or audio endpoints."""
        return ["/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/images/generations", "/v1/messages"]

    def get_supported_apis(self) -> List[str]:
        """Google exposes OpenAI APIs natively and Anthropic via adapter mode."""
        return ["openai", "anthropic"]

    def get_anthropic_mode_for_model(self, model_name: str) -> str:
        if not self._is_chat_capable_model(model_name):
            return "unsupported"
        return "adapter"

    def get_anthropic_request_metadata(self, request, anthropic_beta=None) -> AnthropicRequestMetadata:
        _, metadata = prepare_anthropic_adapter_request(
            request,
            transport="chat",
            allow_thinking=False,
            anthropic_beta=anthropic_beta,
        )
        return metadata

    async def get_available_models(self) -> List[ModelInfo]:
        """Get available Google models using OpenAI-compatible API."""
        try:
            models_response = await self.client.models.list()
            models: List[ModelInfo] = []
            
            for model in models_response.data:
                # Extract model name (remove models/ prefix if present)
                model_name: str = model.id
                if model_name.startswith('models/'):
                    model_name = model_name[7:]
                
                model_info: ModelInfo = self.create_model_info(model_name, "google")
                models.append(model_info)
            
            #print(f"[GOOGLE_PROVIDER] Found {len(models)} available models")
            return models
            
        except Exception as e:
            #print(f"[GOOGLE_PROVIDER] Error fetching models: {e}")
            return []

    async def anthropic_messages(self, request, anthropic_beta=None) -> dict:
        """Serve Anthropic Messages by adapting onto Google's chat-completions API."""
        return await anthropic_adapter_messages(
            self,
            request,
            transport="chat",
            allow_thinking=False,
            anthropic_beta=anthropic_beta,
        )

    async def anthropic_messages_stream(self, request, anthropic_beta=None):
        """Stream Anthropic Messages by adapting onto Google's chat-completions API."""
        async for chunk in anthropic_adapter_messages_stream(
            self,
            request,
            transport="chat",
            allow_thinking=False,
            anthropic_beta=anthropic_beta,
        ):
            yield chunk
