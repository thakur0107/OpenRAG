"""Chat model factory."""

from langchain_core.language_models import BaseChatModel


def build_llm(provider: str, model: str, api_key: str, temperature: float = 0.0) -> BaseChatModel:
    if provider == "Anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model, api_key=api_key, temperature=temperature, max_tokens=2048, streaming=True
        )
    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=api_key, temperature=temperature, streaming=True)
    raise ValueError(f"Unknown provider: {provider}")
