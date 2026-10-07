# ABOUTME: Tests for the LLM provider layer: fixture provider, the live-provider stub, and provider selection.
# ABOUTME: Includes checks that the stub makes no network call and that the provider module imports no SDK or HTTP client.
import ast
import json
import os
import socket

import pytest

from ap_invoice_processor.llm import provider as provider_module
from ap_invoice_processor.llm.provider import (
    AnthropicProvider,
    FixtureProvider,
    LLMProvider,
    LLMProviderError,
    get_provider,
)


def _write(tmp_path, task, doc_id, payload):
    (tmp_path / task).mkdir(exist_ok=True)
    (tmp_path / task / f"{doc_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_fixture_provider_returns_response_and_strips_metadata(tmp_path):
    _write(tmp_path, "extract", "doc-1", {"_authored_from": "reader_text", "total": 5, "vendor_name": "X"})
    out = FixtureProvider(str(tmp_path)).complete("extract", "ignored prompt", "doc-1")
    assert out == {"total": 5, "vendor_name": "X"}


def test_fixture_provider_ignores_prompt_content(tmp_path):
    _write(tmp_path, "gl", "doc-1", {"lines": []})
    p = FixtureProvider(str(tmp_path))
    assert p.complete("gl", "prompt A", "doc-1") == p.complete("gl", "prompt B", "doc-1")


def test_fixture_provider_missing_fixture_is_a_clear_error(tmp_path):
    with pytest.raises(LLMProviderError, match="no fixture for task 'extract', document 'nope'"):
        FixtureProvider(str(tmp_path)).complete("extract", "p", "nope")


@pytest.mark.parametrize("task,doc_id", [("extract", "../secret"), ("../x", "doc"), ("extract", "a/b"), ("", "doc"), ("extract", "")])
def test_fixture_provider_rejects_path_traversal(tmp_path, task, doc_id):
    with pytest.raises(LLMProviderError, match="invalid"):
        FixtureProvider(str(tmp_path)).complete(task, "p", doc_id)


def test_fixture_provider_rejects_non_object_and_unreadable_fixtures(tmp_path):
    _write(tmp_path, "extract", "list", [1, 2])
    (tmp_path / "extract" / "broken.json").write_text("{not json", encoding="utf-8")
    p = FixtureProvider(str(tmp_path))
    with pytest.raises(LLMProviderError, match="JSON object"):
        p.complete("extract", "p", "list")
    with pytest.raises(LLMProviderError, match="unreadable fixture"):
        p.complete("extract", "p", "broken")


def test_providers_satisfy_the_protocol(tmp_path):
    assert isinstance(FixtureProvider(str(tmp_path)), LLMProvider)
    assert isinstance(AnthropicProvider(), LLMProvider)


def test_anthropic_stub_raises_with_and_without_an_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    with pytest.raises(NotImplementedError, match="spend cap"):
        AnthropicProvider().complete("extract", "p", "doc-1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(NotImplementedError, match="spend cap"):
        AnthropicProvider().complete("extract", "p", "doc-1")


def test_anthropic_stub_makes_no_network_call(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    with pytest.raises(NotImplementedError):
        AnthropicProvider().complete("gl", "p", "doc-1")


def test_provider_module_imports_no_sdk_or_http_client():
    with open(provider_module.__file__, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    banned = {"anthropic", "requests", "httpx", "urllib", "urllib3", "http", "socket", "aiohttp", "google"}
    assert not imported & banned, imported & banned


def test_get_provider_defaults_to_fixture(monkeypatch, tmp_path):
    monkeypatch.delenv("AP_LLM_PROVIDER", raising=False)
    assert isinstance(get_provider(fixtures_dir=str(tmp_path)), FixtureProvider)


def test_get_provider_reads_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AP_LLM_PROVIDER", "anthropic")
    assert isinstance(get_provider(), AnthropicProvider)
    monkeypatch.setenv("AP_LLM_PROVIDER", "FIXTURE")
    assert isinstance(get_provider(fixtures_dir=str(tmp_path)), FixtureProvider)


def test_get_provider_explicit_name_beats_env(monkeypatch):
    monkeypatch.setenv("AP_LLM_PROVIDER", "anthropic")
    assert isinstance(get_provider("fixture"), FixtureProvider)


def test_get_provider_unknown_name_is_an_error(monkeypatch):
    monkeypatch.delenv("AP_LLM_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="unknown LLM provider 'gpt'"):
        get_provider("gpt")


def test_default_fixture_dir_points_at_the_repo_fixtures(monkeypatch):
    monkeypatch.delenv("AP_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("AP_LLM_FIXTURES_DIR", raising=False)
    assert get_provider().fixtures_dir == provider_module.DEFAULT_FIXTURES_DIR
    assert os.path.isdir(provider_module.DEFAULT_FIXTURES_DIR)


def test_fixtures_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv("AP_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("AP_LLM_FIXTURES_DIR", str(tmp_path))
    assert get_provider().fixtures_dir == str(tmp_path)
