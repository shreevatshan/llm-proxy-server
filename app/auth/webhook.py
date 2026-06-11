"""Generic webhook utilities for sending notifications."""

import logging
from typing import Optional, Dict, Any
import httpx
from datetime import datetime

from app.config import config

logger = logging.getLogger(__name__)


async def send_notification_webhook(
    event: str,
    data: Dict[str, Any],
    event_context: Optional[str] = None
) -> bool:
    """
    Send a generic notification webhook with event data.

    Args:
        event: The event name (e.g., "user_signup", "api_key_created", "model_updated")
        data: Dictionary containing event-specific data
        event_context: Optional context string for logging purposes

    Returns:
        bool: True if webhook was sent successfully, False otherwise
    """
    webhook_url = config.webhook.notification_webhook_url

    if not webhook_url:
        logger.debug(f"Notification webhook URL not configured, skipping webhook for event: {event}")
        return False

    payload = {
        "event": event,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data
    }

    try:
        # Configure client with more lenient settings for container environments
        async with httpx.AsyncClient(
            timeout=10.0,
            verify=True,  # Verify SSL certificates
            follow_redirects=True  # Follow redirects
        ) as client:
            logger.debug(f"Sending notification webhook for event '{event}' to {webhook_url}")
            logger.debug(f"Payload: {payload}")

            response = await client.post(webhook_url, json=payload)

            logger.debug(f"Webhook response status: {response.status_code}")
            logger.debug(f"Webhook response body: {response.text[:200]}")

            response.raise_for_status()
            context_msg = f" ({event_context})" if event_context else ""
            logger.info(f"Notification webhook sent successfully for event: {event}{context_msg}")
            return True
    except httpx.TimeoutException as e:
        logger.error(f"Notification webhook timeout for event {event}: {webhook_url}")
        logger.error(f"Timeout details: {e}")
        return False
    except httpx.HTTPStatusError as e:
        logger.error(f"Notification webhook HTTP status error for event {event}: {e.response.status_code}")
        logger.error(f"Response body: {e.response.text[:500]}")
        return False
    except httpx.RequestError as e:
        logger.error(f"Notification webhook request error for event {event}: {type(e).__name__}: {e}")
        logger.error(f"URL: {webhook_url}")
        return False
    except Exception as e:
        logger.error(f"Notification webhook unexpected error for event {event}: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


async def send_signup_webhook(
    username: str,
    email: str,
    signup_mode: str,
    user_id: Optional[int] = None,
    is_pending: bool = False,
    oauth_provider: Optional[str] = None
) -> bool:
    """
    Send a webhook notification when a new user signs up.
    Sends in {"text": "..."} format for messaging platform compatibility.

    Args:
        username: The username of the new user
        email: The email of the new user
        signup_mode: The mode of signup ("manual", "oauth", "admin_created")
        user_id: The database ID of the user (if available)
        is_pending: Whether the user account is pending approval
        oauth_provider: The OAuth provider if signup_mode is "oauth"

    Returns:
        bool: True if webhook was sent successfully, False otherwise
    """
    webhook_url = config.webhook.notification_webhook_url

    if not webhook_url:
        logger.debug("Notification webhook URL not configured, skipping signup webhook")
        return False

    # Build human-readable message
    signup_type = {
        "manual": "Manual Signup",
        "oauth": f"OAuth Signup ({oauth_provider})" if oauth_provider else "OAuth Signup",
        "admin_created": "Admin Created"
    }.get(signup_mode, signup_mode)

    status = "⏳ Pending Approval" if is_pending else "✅ Active"

    text = f"""🔔 New User Signup

👤 **Username:** {username}
📧 **Email:** {email}
🔐 **Signup Method:** {signup_type}
📊 **Status:** {status}
🆔 **User ID:** {user_id}
🌐 **Domain:** {config.server.domain}"""

    payload = {"text": text}

    try:
        # Configure client with more lenient settings for container environments
        async with httpx.AsyncClient(
            timeout=10.0,
            verify=True,  # Verify SSL certificates
            follow_redirects=True  # Follow redirects
        ) as client:
            logger.debug(f"Sending signup webhook to {webhook_url}")
            logger.debug(f"Payload: {payload}")

            response = await client.post(webhook_url, json=payload)

            logger.debug(f"Webhook response status: {response.status_code}")
            logger.debug(f"Webhook response body: {response.text[:200]}")

            response.raise_for_status()
            logger.info(f"Signup webhook sent successfully for user {username} (mode: {signup_mode})")
            return True
    except httpx.TimeoutException as e:
        logger.error(f"Signup webhook timeout for user {username}: {webhook_url}")
        logger.error(f"Timeout details: {e}")
        return False
    except httpx.HTTPStatusError as e:
        logger.error(f"Signup webhook HTTP status error for user {username}: {e.response.status_code}")
        logger.error(f"Response body: {e.response.text[:500]}")
        return False
    except httpx.RequestError as e:
        logger.error(f"Signup webhook request error for user {username}: {type(e).__name__}: {e}")
        logger.error(f"URL: {webhook_url}")
        return False
    except Exception as e:
        logger.error(f"Signup webhook unexpected error for user {username}: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
