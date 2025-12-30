from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama
from .settings import settings


def create_llm(streaming: bool = False, max_tokens: int = None, temperature: float = None, **kwargs):
    provider = settings.get_active_provider()
    temp = temperature or settings.llm_temperature
    max_tok = max_tokens or settings.llm_max_tokens
    timeout = kwargs.pop("timeout", settings.llm_timeout)

    if provider == "anthropic":
        return ChatAnthropic(
            model=settings.llm_model,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=temp,
            max_tokens=max_tok,
            streaming=streaming,
            timeout=timeout,
            **kwargs
        )
    else:
        return ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            temperature=temp,
            max_tokens=max_tok,
            streaming=streaming,
            timeout=timeout,
            **kwargs
        )
"""
    else:
        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=temp,
            num_predict=max_tok,
            **kwargs
        )
"""