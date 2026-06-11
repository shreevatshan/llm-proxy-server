"""
Request/Response Transformation System

This module provides a modular system for preprocessing requests and post-processing responses
with easy-to-add and remove processors.
"""

from typing import Dict, Any, Optional, List, Type
from abc import ABC, abstractmethod
from pydantic import BaseModel
from app.openai_models import (
    ChatCompletionRequest, 
    CompletionRequest, 
    ChatCompletionResponse, 
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageResponse,
    AudioSpeechRequest,
    AudioTranscriptionRequest,
    AudioTranslationRequest,
    AudioTranscriptionResponse,
    AudioTranslationResponse
)


class ProcessorConfig(BaseModel):
    """Base configuration for processors."""
    name: str
    enabled: bool = True
    priority: int = 100  # Lower numbers execute first
    config: Dict[str, Any] = {}


class TransformationConfig(BaseModel):
    """Configuration for request/response transformations."""
    enabled: bool = True
    request_processors: List[ProcessorConfig] = []
    response_processors: List[ProcessorConfig] = []


class BaseProcessor(ABC):
    """Abstract base class for all processors."""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
        self.priority = config.priority
        self.processor_config = config.config
    
    @abstractmethod
    async def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """Process the data with the given context."""
        pass
    
    def is_applicable(self, data: Any) -> bool:
        """Check if this processor is applicable to the given data type."""
        return True


class RequestProcessor(BaseProcessor):
    """Base class for request preprocessing."""
    
    async def process(self, request: Any, context: Dict[str, Any]) -> Any:
        """Preprocess the request."""
        return request


class ResponseProcessor(BaseProcessor):
    """Base class for response post-processing."""
    
    async def process(self, response: Any, context: Dict[str, Any]) -> Any:
        """Post-process the response."""
        return response


class ProcessorRegistry:
    """Registry for managing available processors."""
    
    def __init__(self):
        self._request_processors: Dict[str, Type[RequestProcessor]] = {}
        self._response_processors: Dict[str, Type[ResponseProcessor]] = {}
    
    def register_request_processor(self, name: str, processor_class: Type[RequestProcessor]):
        """Register a request processor class."""
        self._request_processors[name] = processor_class
    
    def register_response_processor(self, name: str, processor_class: Type[ResponseProcessor]):
        """Register a response processor class."""
        self._response_processors[name] = processor_class
    
    def get_request_processor(self, name: str) -> Optional[Type[RequestProcessor]]:
        """Get a request processor class by name."""
        return self._request_processors.get(name)
    
    def get_response_processor(self, name: str) -> Optional[Type[ResponseProcessor]]:
        """Get a response processor class by name."""
        return self._response_processors.get(name)
    
    def list_request_processors(self) -> List[str]:
        """List all registered request processor names."""
        return list(self._request_processors.keys())
    
    def list_response_processors(self) -> List[str]:
        """List all registered response processor names."""
        return list(self._response_processors.keys())


class TransformationManager:
    """Manages request/response transformations with modular processors."""
    
    def __init__(self, config: TransformationConfig, registry: ProcessorRegistry):
        self.config = config
        self.registry = registry
        self.request_processors: List[RequestProcessor] = []
        self.response_processors: List[ResponseProcessor] = []
        
        # Initialize processors from configuration
        self._initialize_processors()
    
    def _initialize_processors(self):
        """Initialize processors from configuration."""
        # Initialize request processors
        for processor_config in self.config.request_processors:
            if not processor_config.enabled:
                continue
            
            processor_class = self.registry.get_request_processor(processor_config.name)
            if processor_class:
                try:
                    processor = processor_class(processor_config)
                    self.request_processors.append(processor)
                except Exception as e:
                    print(f"Failed to initialize request processor '{processor_config.name}': {e}")
        
        # Sort by priority (lower numbers first)
        self.request_processors.sort(key=lambda p: p.priority)
        
        # Initialize response processors
        for processor_config in self.config.response_processors:
            if not processor_config.enabled:
                continue
            
            processor_class = self.registry.get_response_processor(processor_config.name)
            if processor_class:
                try:
                    processor = processor_class(processor_config)
                    self.response_processors.append(processor)
                except Exception as e:
                    print(f"Failed to initialize response processor '{processor_config.name}': {e}")
        
        # Sort by priority (lower numbers first)
        self.response_processors.sort(key=lambda p: p.priority)
    
    def add_request_processor(self, processor: RequestProcessor):
        """Add a request processor at runtime."""
        self.request_processors.append(processor)
        self.request_processors.sort(key=lambda p: p.priority)
    
    def add_response_processor(self, processor: ResponseProcessor):
        """Add a response processor at runtime."""
        self.response_processors.append(processor)
        self.response_processors.sort(key=lambda p: p.priority)
    
    def remove_request_processor(self, name: str) -> bool:
        """Remove a request processor by name."""
        for i, processor in enumerate(self.request_processors):
            if processor.name == name:
                del self.request_processors[i]
                return True
        return False
    
    def remove_response_processor(self, name: str) -> bool:
        """Remove a response processor by name."""
        for i, processor in enumerate(self.response_processors):
            if processor.name == name:
                del self.response_processors[i]
                return True
        return False
    
    def list_active_processors(self) -> Dict[str, List[str]]:
        """List all active processors."""
        return {
            "request_processors": [p.name for p in self.request_processors if p.enabled],
            "response_processors": [p.name for p in self.response_processors if p.enabled]
        }
    
    async def preprocess_request(self, request: Any, context: Dict[str, Any] = None) -> Any:
        """Apply all request preprocessors."""
        if not self.config.enabled:
            return request
        
        if context is None:
            context = {}
        
        processed_request = request
        for processor in self.request_processors:
            if not processor.enabled or not processor.is_applicable(processed_request):
                continue
            
            try:
                processed_request = await processor.process(processed_request, context)
            except Exception as e:
                print(f"Request processor '{processor.name}' failed: {e}")
                # Continue with the current state if a processor fails
                continue
        
        return processed_request
    
    async def postprocess_response(self, response: Any, context: Dict[str, Any] = None) -> Any:
        """Apply all response postprocessors."""
        if not self.config.enabled:
            return response
        
        if context is None:
            context = {}
        
        processed_response = response
        for processor in self.response_processors:
            if not processor.enabled or not processor.is_applicable(processed_response):
                continue
            
            try:
                processed_response = await processor.process(processed_response, context)
            except Exception as e:
                print(f"Response processor '{processor.name}' failed: {e}")
                # Continue with the current state if a processor fails
                continue
        
        return processed_response


# Built-in processors

class HeaderInjectionProcessor(RequestProcessor):
    """Injects custom headers into requests."""
    
    def is_applicable(self, data: Any) -> bool:
        """Apply to all request types."""
        return True
    
    async def process(self, request: Any, context: Dict[str, Any]) -> Any:
        """Inject headers based on configuration."""
        headers_to_inject = self.processor_config.get("headers", {})
        
        # Add headers to context for downstream processing
        if "headers" not in context:
            context["headers"] = {}
        context["headers"].update(headers_to_inject)
        
        return request


class ModelMappingProcessor(RequestProcessor):
    """Maps model names to different values."""
    
    def is_applicable(self, data: Any) -> bool:
        """Apply to requests with model field."""
        return hasattr(data, 'model')
    
    async def process(self, request: Any, context: Dict[str, Any]) -> Any:
        """Map model names based on configuration."""
        model_mappings = self.processor_config.get("mappings", {})
        
        if hasattr(request, 'model') and request.model in model_mappings:
            original_model = request.model
            request.model = model_mappings[original_model]
            
            # Store original model in context
            context["original_model"] = original_model
        
        return request


class ResponseMetadataProcessor(ResponseProcessor):
    """Adds metadata to responses."""
    
    def is_applicable(self, data: Any) -> bool:
        """Apply to response objects."""
        return isinstance(data, (ChatCompletionResponse, CompletionResponse, EmbeddingResponse))
    
    async def process(self, response: Any, context: Dict[str, Any]) -> Any:
        """Add metadata to responses."""
        metadata = self.processor_config.get("metadata", {})
        
        # Add metadata if the response supports it
        if hasattr(response, 'metadata'):
            if response.metadata is None:
                response.metadata = {}
            response.metadata.update(metadata)
        
        return response


class ContentFilterProcessor(RequestProcessor):
    """Filters or modifies content based on rules."""
    
    def is_applicable(self, data: Any) -> bool:
        """Apply to chat completion requests."""
        return isinstance(data, ChatCompletionRequest)
    
    async def process(self, request: ChatCompletionRequest, context: Dict[str, Any]) -> ChatCompletionRequest:
        """Filter content based on configuration."""
        filter_rules = self.processor_config.get("rules", [])
        
        for message in request.messages:
            for rule in filter_rules:
                if rule.get("type") == "replace":
                    pattern = rule.get("pattern", "")
                    replacement = rule.get("replacement", "")
                    if pattern and isinstance(message.content, str):
                        import re
                        message.content = re.sub(pattern, replacement, message.content)
        
        return request


# Global instances
_processor_registry: Optional[ProcessorRegistry] = None
_transformation_manager: Optional[TransformationManager] = None


def get_processor_registry() -> ProcessorRegistry:
    """Get the global processor registry instance."""
    global _processor_registry
    if _processor_registry is None:
        _processor_registry = ProcessorRegistry()
        # Register built-in processors
        _processor_registry.register_request_processor("header_injection", HeaderInjectionProcessor)
        _processor_registry.register_request_processor("model_mapping", ModelMappingProcessor)
        _processor_registry.register_request_processor("content_filter", ContentFilterProcessor)
        _processor_registry.register_response_processor("response_metadata", ResponseMetadataProcessor)
    
    return _processor_registry


def get_transformation_manager() -> Optional[TransformationManager]:
    """Get the global transformation manager instance."""
    return _transformation_manager


def initialize_transformation_manager(config: TransformationConfig):
    """Initialize the global transformation manager."""
    global _transformation_manager
    registry = get_processor_registry()
    _transformation_manager = TransformationManager(config, registry)


def create_default_transformation_config() -> TransformationConfig:
    """Create a default transformation configuration."""
    return TransformationConfig(
        enabled=True,
        request_processors=[],
        response_processors=[]
    )


# Utility functions for dynamic processor management

def add_processor_at_runtime(processor_type: str, name: str, processor_class: Type[BaseProcessor]):
    """Add a new processor type at runtime."""
    registry = get_processor_registry()
    
    if processor_type == "request":
        if not issubclass(processor_class, RequestProcessor):
            raise ValueError("Processor must inherit from RequestProcessor")
        registry.register_request_processor(name, processor_class)
    elif processor_type == "response":
        if not issubclass(processor_class, ResponseProcessor):
            raise ValueError("Processor must inherit from ResponseProcessor")
        registry.register_response_processor(name, processor_class)
    else:
        raise ValueError("processor_type must be 'request' or 'response'")


def create_processor_instance(processor_type: str, name: str, config: Dict[str, Any]) -> Optional[BaseProcessor]:
    """Create a processor instance from configuration."""
    registry = get_processor_registry()
    
    processor_config = ProcessorConfig(
        name=f"{name}_instance",
        enabled=config.get("enabled", True),
        priority=config.get("priority", 100),
        config=config.get("config", {})
    )
    
    if processor_type == "request":
        processor_class = registry.get_request_processor(name)
        if processor_class:
            return processor_class(processor_config)
    elif processor_type == "response":
        processor_class = registry.get_response_processor(name)
        if processor_class:
            return processor_class(processor_config)
    
    return None
