"""ZOHO OAuth integration for authentication."""

import os
import base64
import json
import httpx
from jose import jwt, JWTError
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import HTTPException, status
from urllib.parse import urlencode
from pydantic import BaseModel


class ZohoConfig:
    """ZOHO OAuth configuration."""

    def __init__(self):
        from app.config import config
        self.client_id = config.oauth.zoho_client_id
        self.client_secret = config.oauth.zoho_client_secret
        self.redirect_uri = config.oauth.zoho_redirect_uri
        self.scope = "email profile openid"
        
    def is_configured(self) -> bool:
        """Check if ZOHO OAuth is properly configured."""
        return bool(self.client_id and self.client_secret)


class ZohoUserInfo(BaseModel):
    """ZOHO user information from ID token."""
    sub: str  # Unique user identifier
    email: str
    email_verified: Optional[bool] = None
    name: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    picture: Optional[str] = None  
    iss: str
    aud: str
    exp: int
    iat: int


class ZohoOAuth:
    """ZOHO OAuth handler."""
    
    def __init__(self):
        self.config = ZohoConfig()
        self._cached_base_url: Optional[str] = None
        self._cached_locations: Optional[Dict[str, str]] = None
        if not self.config.is_configured():
            raise ValueError("ZOHO OAuth is not properly configured. Please set ZOHO_CLIENT_ID and ZOHO_CLIENT_SECRET environment variables.")
    
    async def _fetch_server_info(self) -> Dict[str, str]:
        """Fetch ZOHO server info with all datacenter locations.
        
        Returns a dict mapping location codes to their base URLs.
        Example: {'us': 'https://accounts.zoho.com', 'in': 'https://accounts.zoho.in', ...}
        """
        if self._cached_locations:
            return self._cached_locations
            
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("https://accounts.zoho.com/oauth/serverinfo")
                response.raise_for_status()
                server_info = response.json()
                
                # The response contains "locations" with all available data centers
                locations = server_info.get("locations", {})
                
                if locations:
                    self._cached_locations = locations
                    return locations
                else:
                    # Fallback if no locations in response
                    return {"us": "https://accounts.zoho.com"}
                    
        except Exception as e:
            # Fallback to default US datacenter on error
            return {"us": "https://accounts.zoho.com"}
    
    async def _get_base_url(self, location: Optional[str] = None) -> str:
        """Get ZOHO base URL for a specific location or default.
        
        Args:
            location: Optional location code (e.g., 'us', 'in', 'eu')
            
        Returns:
            Base URL for the specified or default datacenter
        """
        locations = await self._fetch_server_info()
        
        if location:
            # Return URL for specified location, fallback to US if not found
            return locations.get(location, locations.get("us", "https://accounts.zoho.com"))
        
        # Default to US datacenter
        if not self._cached_base_url:
            self._cached_base_url = locations.get("us", "https://accounts.zoho.com")
        
        return self._cached_base_url
    
    async def get_authorization_url(self, state: Optional[str] = None) -> str:
        """Generate ZOHO OAuth authorization URL."""
        base_url = await self._get_base_url()
        
        params = {
            "client_id": self.config.client_id,
            "response_type": "code",
            "scope": self.config.scope,
            "redirect_uri": self.config.redirect_uri,
            "access_type": "offline"
        }
        
        if state:
            params["state"] = state
            
        return f"{base_url}/oauth/v2/auth?{urlencode(params)}"
    
    async def exchange_code_for_token(self, authorization_code: str, location: Optional[str] = None) -> Dict[str, Any]:
        """Exchange authorization code for ID token and access token.
        
        Args:
            authorization_code: The OAuth authorization code
            location: Optional datacenter location code (e.g., 'us', 'in', 'eu')
                     If provided, uses that datacenter; otherwise defaults to US
        """
        # Get base URL dynamically from server info
        # Location parameter allows override to specific datacenter
        base_url = await self._get_base_url(location)
        token_url = f"{base_url}/oauth/v2/token"
        
        data = {
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "grant_type": "authorization_code",
            "code": authorization_code,
            "redirect_uri": self.config.redirect_uri
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(
                    token_url,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to exchange authorization code: {e.response.text}"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"OAuth token exchange failed: {str(e)}"
                )
    
    async def decode_id_token(self, id_token: str) -> ZohoUserInfo:
        """Decode and validate ZOHO ID token (JWT)."""
        try:
            # Manually decode JWT payload without verification
            # JWT format: header.payload.signature
            parts = id_token.split('.')
            if len(parts) != 3:
                raise ValueError("Invalid JWT format")
            
            # Decode the payload (second part)
            payload = parts[1]
            # Add padding if needed for base64 decoding
            missing_padding = len(payload) % 4
            if missing_padding:
                payload += '=' * (4 - missing_padding)
            
            decoded_bytes = base64.urlsafe_b64decode(payload)
            decoded_token = json.loads(decoded_bytes.decode('utf-8'))
            
            # Validate required fields
            required_fields = ["sub", "email", "iss", "aud", "exp", "iat"]
            for field in required_fields:
                if field not in decoded_token:
                    raise ValueError(f"Missing required field: {field}")
            
            # Set default name if not present
            if "name" not in decoded_token:
                decoded_token["name"] = decoded_token.get("email", "Unknown User")
            
            # Validate issuer dynamically against fetched datacenters
            # The issuer can be in format "accounts.zoho.com" or "https://accounts.zoho.com"
            issuer = decoded_token["iss"]
            locations = await self._fetch_server_info()
            
            # Build valid issuers from fetched locations (both with and without https://)
            valid_issuers = set()
            for base_url in locations.values():
                # Extract domain from URL (e.g., "https://accounts.zoho.com" -> "accounts.zoho.com")
                domain = base_url.replace("https://", "").replace("http://", "")
                valid_issuers.add(domain)
                valid_issuers.add(base_url)
            
            if issuer not in valid_issuers:
                raise ValueError(f"Invalid issuer: {issuer}")
            
            # Validate audience (client_id)
            if decoded_token["aud"] != self.config.client_id:
                raise ValueError(f"Invalid audience: {decoded_token['aud']}")
            
            # Check if token is expired
            if decoded_token["exp"] < datetime.utcnow().timestamp():
                raise ValueError("Token has expired")
            
            return ZohoUserInfo(**decoded_token)
            
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid ID token: {str(e)}"
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Token validation failed: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to decode ID token: {str(e)}"
            )
    
    async def get_user_info(self, authorization_code: str, location: Optional[str] = None) -> ZohoUserInfo:
        """Get user information from ZOHO OAuth."""
        # Exchange code for tokens
        token_response = await self.exchange_code_for_token(authorization_code, location)
        
        # Extract and decode ID token
        id_token = token_response.get("id_token")
        if not id_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No ID token received from ZOHO"
            )
        
        return await self.decode_id_token(id_token)


# Global ZOHO OAuth instance
def get_zoho_oauth() -> Optional[ZohoOAuth]:
    """Get ZOHO OAuth instance if configured."""
    try:
        return ZohoOAuth()
    except ValueError:
        # ZOHO OAuth not configured
        return None


zoho_oauth = get_zoho_oauth()
