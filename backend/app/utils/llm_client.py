"""
LLM client wrapper
Uses Anthropic SDK directly with OAuth token support (sk-ant-oat...) or API key.
"""

import json
import re
from typing import Optional, Dict, Any, List

from ..config import Config
from .anthropic_client import AnthropicClient, load_anthropic_key


class LLMClient:
    """LLM client — backed by Anthropic SDK, supports OAuth tokens."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,  # kept for interface compatibility, not used
        model: Optional[str] = None
    ):
        resolved_key = api_key or load_anthropic_key()
        if not resolved_key:
            raise ValueError(
                "Anthropic key not configured. Set ANTHROPIC_KEY env var or "
                "POST to /api/setup/key with {\"key\": \"sk-ant-...\"}"
            )
        self.model = model or Config.LLM_MODEL_NAME
        self._impl = AnthropicClient(key=resolved_key, model=self.model)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        Send a chat request

        Args:
            messages: List of messages
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            response_format: Response format (e.g., JSON mode)

        Returns:
            Model response text
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        return self._impl.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Send a chat request and return JSON

        Args:
            messages: List of messages
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens

        Returns:
            Parsed JSON object
        """
        return self._impl.chat_json(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

