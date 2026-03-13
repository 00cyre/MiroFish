"""
Anthropic client with OAuth token support.

Mirrors the pattern from claudetree's analyzer.ts:
  - If the key starts with 'sk-ant-oat' it is a Claude OAuth token
    and we use authToken + the oauth-2025-04-20 beta header.
  - Otherwise it is a regular API key.

The key is read from (in priority order):
  1. ANTHROPIC_KEY env var
  2. ~/.mirofish/config.json  { "anthropicKey": "..." }
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic

# ---------------------------------------------------------------------------
# Local config helpers
# ---------------------------------------------------------------------------

CONFIG_DIR = Path.home() / ".mirofish"
CONFIG_FILE = CONFIG_DIR / "config.json"


def _load_local_config() -> Dict[str, Any]:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_anthropic_key(key: str) -> None:
    """Persist the Anthropic key (API or OAuth) to ~/.mirofish/config.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _load_local_config()
    cfg["anthropicKey"] = key
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def load_anthropic_key() -> Optional[str]:
    """Load the Anthropic key from env or local config."""
    key = os.environ.get("ANTHROPIC_KEY")
    if key:
        return key
    return _load_local_config().get("anthropicKey")


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def _is_oauth_token(key: str) -> bool:
    return key.startswith("sk-ant-oat")


def create_anthropic_client(key: Optional[str] = None) -> anthropic.Anthropic:
    """
    Return an Anthropic client configured for API key or OAuth token.

    Args:
        key: Optional override. Falls back to load_anthropic_key().

    Raises:
        ValueError: If no key is available.
    """
    resolved = key or load_anthropic_key()
    if not resolved:
        raise ValueError(
            "No Anthropic key found. Set ANTHROPIC_KEY env var or save one with "
            "save_anthropic_key('sk-ant-...')."
        )

    if _is_oauth_token(resolved):
        return anthropic.Anthropic(
            auth_token=resolved,
            api_key=None,           # must be explicitly None for OAuth
            default_headers={"anthropic-beta": "oauth-2025-04-20"},
        )

    return anthropic.Anthropic(api_key=resolved)


# ---------------------------------------------------------------------------
# High-level chat wrapper (matches LLMClient interface)
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-sonnet-4-5"


class AnthropicClient:
    """
    Drop-in replacement for LLMClient that uses the Anthropic SDK directly,
    supporting both API keys and Claude OAuth tokens.
    """

    def __init__(
        self,
        key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self._client = create_anthropic_client(key)
        self.model = model or os.environ.get("LLM_MODEL_NAME", DEFAULT_MODEL)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,  # accepted but ignored (Anthropic handles JSON via prompt)
    ) -> str:
        """Send a chat request and return the response text."""
        # Separate system prompt if present
        system = None
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append({"role": m["role"], "content": m["content"]})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": filtered,
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = self._client.messages.create(**kwargs)
        content = response.content[0].text
        # Strip any <think>…</think> reasoning blocks (some models emit these)
        content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
        return content

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """Send a chat request and return a parsed JSON dict."""
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        cleaned = response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON returned by LLM: {cleaned}")
