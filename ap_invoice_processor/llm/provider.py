# ABOUTME: LLMProvider protocol, the fixture-backed implementation, and provider selection (fixture or anthropic).
# ABOUTME: Provider choice comes from AP_LLM_PROVIDER (default fixture); the live provider is imported only when chosen.
import json
import os
import re
from typing import Any, Dict, Optional, Protocol, runtime_checkable

PROVIDER_ENV = "AP_LLM_PROVIDER"
FIXTURES_DIR_ENV = "AP_LLM_FIXTURES_DIR"
DEFAULT_PROVIDER = "fixture"
DEFAULT_FIXTURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tests", "fixtures", "llm"
)
# Task and document ids become path segments, so they are restricted to a safe alphabet.
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class LLMProviderError(RuntimeError):
    """Raised when a provider cannot return a response (fixture: no fixture or an unreadable one; live: any failed call)."""


class ProviderConfigError(ValueError):
    """Raised when a provider cannot be built from its configuration (missing key or cap, unreadable settings)."""


@runtime_checkable
class LLMProvider(Protocol):
    def complete(self, task: str, prompt: str, doc_id: str) -> Dict[str, Any]:
        """Return the model's structured response (already JSON-decoded) for one task on one document."""
        ...


class FixtureProvider:
    """Serves hand-authored responses from <fixtures_dir>/<task>/<doc_id>.json; the prompt is not consulted."""

    def __init__(self, fixtures_dir: str):
        self.fixtures_dir = fixtures_dir

    def complete(self, task: str, prompt: str, doc_id: str) -> Dict[str, Any]:
        for label, value in (("task", task), ("doc_id", doc_id)):
            if not isinstance(value, str) or not _SAFE_SEGMENT.match(value):
                raise LLMProviderError(f"invalid {label} for fixture lookup: {value!r}")
        path = os.path.join(self.fixtures_dir, task, f"{doc_id}.json")
        if not os.path.isfile(path):
            raise LLMProviderError(f"no fixture for task {task!r}, document {doc_id!r}: expected {path}")
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as exc:
            raise LLMProviderError(f"unreadable fixture {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMProviderError(f"fixture {path} must hold a JSON object, got {type(data).__name__}")
        # Keys starting with an underscore are fixture metadata (for example _authored_from), not response content.
        return {k: v for k, v in data.items() if not k.startswith("_")}


def __getattr__(name: str) -> Any:
    # AnthropicProvider is re-exported lazily so importing this module never loads the Anthropic SDK.
    if name == "AnthropicProvider":
        from ap_invoice_processor.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_provider(
    name: Optional[str] = None, fixtures_dir: Optional[str] = None, max_usd: Optional[float] = None
) -> LLMProvider:
    """Build the provider named by `name`, else by AP_LLM_PROVIDER, else the fixture provider.

    `max_usd` is a cap on one run's spend and applies to the live provider only. Building the live provider raises
    ProviderConfigError (a ValueError) when its key or spend cap is missing.
    """
    chosen = (name or os.environ.get(PROVIDER_ENV) or DEFAULT_PROVIDER).strip().lower()
    if chosen == "fixture":
        return FixtureProvider(fixtures_dir or os.environ.get(FIXTURES_DIR_ENV) or DEFAULT_FIXTURES_DIR)
    if chosen == "anthropic":
        from ap_invoice_processor.llm.anthropic_provider import build_anthropic_provider

        return build_anthropic_provider(max_usd=max_usd)
    raise ValueError(f"unknown LLM provider {chosen!r}; expected 'fixture' or 'anthropic'")
