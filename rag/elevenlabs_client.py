"""
ElevenLabs API client for conversation management.
"""
import httpx
from typing import Any

from rag.config import ELEVENLABS_API_KEY

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1/convai"


def get_all_conversations() -> dict[str, Any]:
    """
    Get all conversations from ElevenLabs API.
    
    Returns:
        Dictionary containing conversations list with conversation_id, message_count, and call_summary_title
    """
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY
    }
    
    try:
        with httpx.Client() as client:
            response = client.get(
                f"{ELEVENLABS_BASE_URL}/conversations",
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract only the fields we need: conversation_id, message_count, and call_summary_title
            conversations = []
            for conv in data.get("conversations", []):
                conversations.append({
                    "conversation_id": conv.get("conversation_id"),
                    "message_count": conv.get("message_count"),
                    "call_summary_title": conv.get("call_summary_title")
                })
            
            return {
                "conversations": conversations,
                "has_more": data.get("has_more", False),
                "next_cursor": data.get("next_cursor")
            }
    except httpx.HTTPStatusError as e:
        raise ValueError(f"ElevenLabs API error: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        raise ValueError(f"Request error: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error: {str(e)}")


def get_conversation_by_id(conversation_id: str) -> dict[str, Any]:
    """
    Get a specific conversation by ID from ElevenLabs API.
    
    Args:
        conversation_id: The ID of the conversation to retrieve
        
    Returns:
        Full conversation data from ElevenLabs API
    """
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY
    }
    
    try:
        with httpx.Client() as client:
            response = client.get(
                f"{ELEVENLABS_BASE_URL}/conversations/{conversation_id}",
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise ValueError(f"ElevenLabs API error: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        raise ValueError(f"Request error: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error: {str(e)}")

