from abc import ABC, abstractmethod
import httpx

from openai import OpenAI
from anthropic import Anthropic
from google import genai

from app.config import get_settings


class LLM(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        pass


class OpenAILLM(LLM):
    def __init__(self, model: str, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        response = self.client.responses.create(
            model=self.model,
            temperature=temperature,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text


class AnthropicLLM(LLM):
    def __init__(self, model: str, api_key: str):
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text


class GeminiLLM(LLM):
    def __init__(self, model: str, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        prompt = f"{system_prompt}\n\n{user_prompt}"
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": temperature},
        )
        return response.text or ""


class OllamaLLM(LLM):
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["message"]["content"]


class EchoLLM(LLM):
    """Useful for local testing without any paid LLM key."""
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        return "Echo test mode: retrieval worked. Add a real LLM API key to generate final answers.\n\n" + user_prompt[:1500]


def get_llm(provider: str | None = None, model: str | None = None) -> LLM:
    settings = get_settings()
    provider = (provider or settings.llm_provider).lower()
    model = model or settings.llm_model

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required. For local no-key testing, set LLM_PROVIDER=echo.")
        return OpenAILLM(model=model, api_key=settings.openai_api_key)

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")
        return AnthropicLLM(model=model, api_key=settings.anthropic_api_key)

    if provider == "gemini":
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required")
        return GeminiLLM(model=model, api_key=settings.google_api_key)

    if provider == "ollama":
        return OllamaLLM(model=model or settings.ollama_model, base_url=settings.ollama_base_url)

    if provider == "echo":
        return EchoLLM()

    raise ValueError(f"Unsupported LLM provider: {provider}")
