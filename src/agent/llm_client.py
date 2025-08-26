from typing import List, Dict, Any, Optional
from openai import OpenAI
import json
import re


class LLMClient:
    """Thin wrapper over an OpenAI-compatible Chat Completions API."""
    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "llama-3.1-8b-instant") -> None:
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
        )
        return resp.choices[0].message.content or ""

    # Convenience method for JSON-mode responses (planner)
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        raw = self.chat(
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        # Be resilient to stray fences or whitespace
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        try:
            return json.loads(cleaned)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON from model: {e}\nRAW:\n{raw}")