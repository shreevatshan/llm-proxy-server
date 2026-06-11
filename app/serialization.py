"""
Serialization utilities for handling Pydantic models and other complex types.

This module provides utilities to safely serialize objects that may contain
Pydantic models, making them compatible with json.dumps() and tracing libraries.
"""

import json
from typing import Any
from pydantic import BaseModel


def pydantic_encoder(obj: Any) -> Any:
    """
    Custom JSON encoder for Pydantic models and other complex types.
    
    This encoder can be used with json.dumps(obj, default=pydantic_encoder)
    to safely serialize objects that contain Pydantic models.
    
    Args:
        obj: Object to encode
        
    Returns:
        JSON-serializable representation of the object
    """
    if isinstance(obj, BaseModel):
        # Use Pydantic's model_dump() or dict() method
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif hasattr(obj, 'dict'):
            return obj.dict()
    
    # Fallback to string representation for other non-serializable types
    return str(obj)


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely serialize an object to JSON, handling Pydantic models.
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments to pass to json.dumps()
        
    Returns:
        JSON string representation of the object
    """
    # Set default to pydantic_encoder if not provided
    if 'default' not in kwargs:
        kwargs['default'] = pydantic_encoder
    
    return json.dumps(obj, **kwargs)


def safe_dict_conversion(obj: Any) -> dict:
    """
    Safely convert an object to a dictionary, handling Pydantic models.
    
    Args:
        obj: Object to convert
        
    Returns:
        Dictionary representation of the object
    """
    if isinstance(obj, BaseModel):
        # Use Pydantic's model_dump() or dict() method
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif hasattr(obj, 'dict'):
            return obj.dict()
    elif isinstance(obj, dict):
        return obj
    
    # For other types, try to convert to dict or return as-is
    try:
        return dict(obj)
    except (TypeError, ValueError):
        return {"value": str(obj)}
