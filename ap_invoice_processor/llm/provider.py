# ABOUTME: LLMProvider protocol with a fixture-backed implementation and a live-provider stub.
# ABOUTME: Provider choice comes from AP_LLM_PROVIDER (fixture|anthropic, default fixture); no network is ever used here.
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
    """Raised when a provider cannot return a response (for the fixture provider: no fixture, or an unreadable one)."""


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


class AnthropicProvider:
    """Placeholder for the live provider. Makes no network call and imports no SDK; it always raises."""

    def complete(self, task: str, prompt: str, doc_id: str) -> Dict[str, Any]:
        raise NotImplementedError(
            "Live LLM calls are not enabled: they await a decision on provider, API key and spend cap. "
            f"Set {PROVIDER_ENV}=fixture to use the hand-authored fixtures."
        )


def get_provider(name: Optional[str] = None, fixtures_dir: Optional[str] = None) -> LLMProvider:
    """Build the provider named by `name`, else by AP_LLM_PROVIDER, else the fixture provider."""
    chosen = (name or os.environ.get(PROVIDER_ENV) or DEFAULT_PROVIDER).strip().lower()
    if chosen == "fixture":
        return FixtureProvider(fixtures_dir or os.environ.get(FIXTURES_DIR_ENV) or DEFAULT_FIXTURES_DIR)
    if chosen == "anthropic":
        return AnthropicProvider()
    raise ValueError(f"unknown LLM provider {chosen!r}; expected 'fixture' or 'anthropic'")
