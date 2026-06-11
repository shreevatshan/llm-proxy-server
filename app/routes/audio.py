from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Depends
from fastapi.responses import Response
from typing import Optional, List, Union
import base64
import io
import time
from app.openai_models import (
    AudioSpeechRequest,
    AudioTranscriptionRequest,
    AudioTranslationRequest,
    AudioTranscriptionResponse,
    AudioTranslationResponse
)
from app.providers.provider_manager import provider_manager
from app.auth.middleware import authenticate_jwt_or_api_key
from app.auth.models import APIKey, User
from app.auth.admin import AdminUser
from app.tracing import create_span, add_span_attributes, set_span_error
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v1/audio/speech")
async def create_speech(
    request: AudioSpeechRequest,
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_jwt_or_api_key)
):
    """
    Generate audio from text using text-to-speech.
    """
    try:
        # Get provider for the model
        provider_name, model_id = provider_manager._parse_model_name(request.model)
        provider = provider_manager._get_provider(provider_name)
        
        # Generate speech
        audio_data = await provider.audio_speech(request)
        
        # Determine content type based on response format
        content_type_map = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm"
        }
        
        content_type = content_type_map.get(request.response_format, "audio/mpeg")
        
        return Response(
            content=audio_data,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}"
            }
        )
        
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Error in speech generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/v1/audio/transcriptions", response_model=AudioTranscriptionResponse)
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0),
    timestamp_granularities: Optional[str] = Form(None),
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_jwt_or_api_key)
):
    """
    Transcribe audio to text.
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Convert to base64 for processing
        file_b64 = base64.b64encode(file_content).decode('utf-8')
        
        # Parse timestamp_granularities if provided
        granularities = None
        if timestamp_granularities:
            try:
                granularities = timestamp_granularities.split(',')
            except:
                granularities = [timestamp_granularities]
        
        # Create request object
        request = AudioTranscriptionRequest(
            file=file_b64,
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=granularities
        )
        
        # Get provider for the model
        provider_name, model_id = provider_manager._parse_model_name(request.model)
        provider = provider_manager._get_provider(provider_name)
        
        # Perform transcription
        result = await provider.audio_transcription(request)
        
        # Handle different response formats
        if response_format == "text":
            return Response(content=result.text, media_type="text/plain")
        elif response_format == "srt":
            # Convert to SRT format
            srt_content = _convert_to_srt(result)
            return Response(content=srt_content, media_type="text/plain")
        elif response_format == "vtt":
            # Convert to VTT format
            vtt_content = _convert_to_vtt(result)
            return Response(content=vtt_content, media_type="text/plain")
        else:
            # JSON format (default and verbose_json)
            return result
        
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Error in audio transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/v1/audio/translations", response_model=AudioTranslationResponse)
async def create_translation(
    file: UploadFile = File(...),
    model: str = Form(...),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0),
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_jwt_or_api_key)
):
    """
    Translate audio to English text.
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Convert to base64 for processing
        file_b64 = base64.b64encode(file_content).decode('utf-8')
        
        # Create request object
        request = AudioTranslationRequest(
            file=file_b64,
            model=model,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature
        )
        
        # Get provider for the model
        provider_name, model_id = provider_manager._parse_model_name(request.model)
        provider = provider_manager._get_provider(provider_name)
        
        # Perform translation
        result = await provider.audio_translation(request)
        
        # Handle different response formats
        if response_format == "text":
            return Response(content=result.text, media_type="text/plain")
        elif response_format == "srt":
            # Convert to SRT format
            srt_content = _convert_to_srt(result)
            return Response(content=srt_content, media_type="text/plain")
        elif response_format == "vtt":
            # Convert to VTT format
            vtt_content = _convert_to_vtt(result)
            return Response(content=vtt_content, media_type="text/plain")
        else:
            # JSON format (default and verbose_json)
            return result
        
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Error in audio translation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def _convert_to_srt(result: AudioTranscriptionResponse) -> str:
    """Convert transcription result to SRT format."""
    if not result.segments:
        return f"1\n00:00:00,000 --> 00:00:10,000\n{result.text}\n"
    
    srt_content = []
    for i, segment in enumerate(result.segments, 1):
        start_time = _seconds_to_srt_time(segment.start)
        end_time = _seconds_to_srt_time(segment.end)
        srt_content.append(f"{i}\n{start_time} --> {end_time}\n{segment.text.strip()}\n")
    
    return "\n".join(srt_content)


def _convert_to_vtt(result: AudioTranscriptionResponse) -> str:
    """Convert transcription result to VTT format."""
    if not result.segments:
        return f"WEBVTT\n\n00:00:00.000 --> 00:00:10.000\n{result.text}\n"
    
    vtt_content = ["WEBVTT\n"]
    for segment in result.segments:
        start_time = _seconds_to_vtt_time(segment.start)
        end_time = _seconds_to_vtt_time(segment.end)
        vtt_content.append(f"{start_time} --> {end_time}\n{segment.text.strip()}\n")
    
    return "\n".join(vtt_content)


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _seconds_to_vtt_time(seconds: float) -> str:
    """Convert seconds to VTT time format (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
