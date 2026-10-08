# ABOUTME: Provider-agnostic LLM layer: provider interface, invoice extraction and GL coding.
# ABOUTME: The fixture provider is the default; the live Anthropic provider runs only when selected and capped.
from typing import Any

from ap_invoice_processor.llm.provider import (
    FixtureProvider,
    LLMProvider,
    LLMProviderError,
    ProviderConfigError,
    get_provider,
)

__all__ = ["AnthropicProvider", "FixtureProvider", "LLMProvider", "LLMProviderError", "ProviderConfigError", "get_provider"]


def __getattr__(name: str) -> Any:
    # AnthropicProvider loads the Anthropic SDK, so it is resolved on first use rather than at package import.
    if name == "AnthropicProvider":
        from ap_invoice_processor.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
