# ABOUTME: Shared pytest setup: registers the `live` marker and keeps every test away from real API credentials.
# ABOUTME: Only tests marked `live` (opt-in, AP_LIVE=1) may see the machine's key file and provider environment.
import pytest

# Variables that choose or configure a provider, or that hold credentials. A developer machine may set any of them.
_PROVIDER_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "AP_LLM_PROVIDER",
    "AP_LLM_MODEL",
    "AP_LLM_SPEND_CAP_USD",
    "AP_LLM_EFFORT",
    "AP_LLM_MAX_TOKENS",
    "AP_OCR_LANGS",
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: calls the real Anthropic API and spends money; runs only when AP_LIVE=1 is set"
    )


@pytest.fixture(autouse=True)
def isolate_provider_credentials(request, monkeypatch, tmp_path_factory):
    """Hide the real key and key file from every test except those marked `live`."""
    if request.node.get_closest_marker("live"):
        return
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AP_INTAKE_ENV_FILE", str(tmp_path_factory.getbasetemp() / "no-such-credentials.env"))
