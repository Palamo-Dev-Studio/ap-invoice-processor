# ABOUTME: Static checks on the Dockerfile, .dockerignore and .gcloudignore: non-root user, pinned install, no secrets in the image.
# ABOUTME: Docker itself is not needed; the ignore files are evaluated with a small matcher that follows .dockerignore rules.
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


def _rules(name):
    return [line.strip() for line in _read(name).splitlines() if line.strip() and not line.startswith("#")]


def _regex(pattern):
    """Translate one .dockerignore pattern to a regex over a slash-separated path relative to the context root."""
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile(f"^{out}(?:/.*)?$")


def ignored(rules, path):
    """True if `path` is excluded: the last matching rule wins, and a leading "!" re-includes."""
    result = False
    for rule in rules:
        negate = rule.startswith("!")
        if _regex(rule.lstrip("!").lstrip("/")).match(path):
            result = not negate
    return result


# --- Dockerfile -------------------------------------------------------------------------------------------------------


def test_the_image_runs_as_a_non_root_user():
    lines = [l.strip() for l in _read("Dockerfile").splitlines()]
    users = [l for l in lines if l.startswith("USER ")]
    assert users and users[-1] != "USER root" and users[-1] != "USER 0"
    assert lines.index(users[-1]) > max(i for i, l in enumerate(lines) if l.startswith("RUN ") or l.startswith("COPY "))


def test_the_base_image_is_the_python_the_suite_runs_on():
    assert re.search(r"^FROM python:3\.13-slim\s*$", _read("Dockerfile"), re.M)


def test_the_ocr_and_pdf_packages_are_installed():
    text = _read("Dockerfile")
    for package in ("poppler-utils", "tesseract-ocr", "tesseract-ocr-spa", "tesseract-ocr-chi-sim", "tesseract-ocr-chi-tra"):
        assert re.search(rf"^\s+{package}\b", text, re.M), package
    assert "--no-install-recommends" in text and "rm -rf /var/lib/apt/lists" in text


def test_dependencies_come_from_the_pinned_requirements_file():
    assert "pip install -r requirements.txt" in _read("Dockerfile")
    for line in _rules("requirements.txt"):
        assert re.match(r"^[A-Za-z0-9_.\-\[\]]+==[0-9][^=<>~!]*$", line), f"{line!r} is not an exact pin"
    assert any(l.startswith("python-multipart==") for l in _rules("requirements.txt")), "uploads need python-multipart pinned"


def test_the_server_listens_on_port_from_the_environment():
    cmd = [l for l in _read("Dockerfile").splitlines() if l.startswith("CMD")]
    assert len(cmd) == 1
    assert "uvicorn web.server:web_app" in cmd[0] and "--host 0.0.0.0" in cmd[0] and "${PORT" in cmd[0]


def test_the_image_bakes_in_no_credential_or_provider_choice():
    text = "\n".join(l for l in _read("Dockerfile").splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"ANTHROPIC|API_KEY|SECRET|TOKEN|PASSWORD", text, re.I)
    assert "AP_LLM_PROVIDER" not in text and "AP_DEMO_MODE" not in text, "those are set per deployment, not in the image"


def test_the_ledger_defaults_to_tmp_in_the_container():
    assert re.search(r"^ENV AP_SPEND_LEDGER_PATH=/tmp/", _read("Dockerfile"), re.M)


def test_the_container_caps_pdf_pages_at_the_dashboard_default():
    from web.uploads import DEFAULT_MAX_PDF_PAGES

    match = re.search(r"^ENV AP_MAX_PDF_PAGES=(\d+)$", _read("Dockerfile"), re.M)
    assert match and int(match.group(1)) == DEFAULT_MAX_PDF_PAGES


# --- ignore files -----------------------------------------------------------------------------------------------------

SECRET_PATHS = [
    ".env",
    "prod.env",
    ".env.local",
    "config/anthropic.env",
    "eval/out/spend.json",
    "eval/out/extraction.csv",
    "tmp/spend.json",
    "deploy/key.pem",
    ".git/config",
    ".claude/settings.json",
    ".venv/lib/python3.13/site.py",
]
KEPT_PATHS = [
    "web/server.py",
    "web/static/app.js",
    "ap_invoice_processor/nodes.py",
    "mcp_server/netsuite_mcp_server.py",
    "skills/ap_invoice_skill/SKILL.md",
    "data/gl_chart_of_accounts.json",
    "data/vendor_master.json",
    "data/synthetic_invoices/invoices.json",
    "data/corpus/pdf/en-001.pdf",
    "data/corpus/images/en-004-scan.png",
    "tests/fixtures/llm/extract/en-001.json",
    "tests/fixtures/llm/gl/en-001.json",
    "requirements.txt",
    "LICENSE",
]


@pytest.mark.parametrize("ignore_file", [".dockerignore", ".gcloudignore"])
@pytest.mark.parametrize("path", SECRET_PATHS)
def test_secrets_and_run_output_are_excluded(ignore_file, path):
    assert ignored(_rules(ignore_file), path), f"{path} would be uploaded by {ignore_file}"


@pytest.mark.parametrize("ignore_file", [".dockerignore", ".gcloudignore"])
@pytest.mark.parametrize("path", KEPT_PATHS)
def test_runtime_files_are_not_excluded(ignore_file, path):
    assert not ignored(_rules(ignore_file), path), f"{path} is needed at runtime but {ignore_file} excludes it"


def test_the_docker_context_also_drops_the_tests_and_the_answer_key():
    rules = _rules(".dockerignore")
    for path in ("tests/test_nodes.py", "tests/conftest.py", "tests/fixtures/ocr/x.png", "data/corpus/ground_truth/en-001.json", "data/corpus/gl_labels.json", "eval/extraction_eval.py", "docs/AUDIT.md", ".github/workflows/ci.yml"):
        assert ignored(rules, path), path


def test_the_gcloud_upload_keeps_the_dockerfile_it_builds_from():
    rules = _rules(".gcloudignore")
    assert not ignored(rules, "Dockerfile") and not ignored(rules, ".dockerignore")


def test_the_matcher_itself_can_fail():
    """Negative control: an ignore file with no rules excludes nothing, and the secret checks above would catch it."""
    assert not ignored([], ".env")
    assert ignored([".env"], ".env") and not ignored([".env"], "web/server.py")
    assert not ignored(["tests/*", "!tests/fixtures", "tests/fixtures/*", "!tests/fixtures/llm"], "tests/fixtures/llm/a.json")
    assert ignored(["tests/*", "!tests/fixtures", "tests/fixtures/*", "!tests/fixtures/llm"], "tests/fixtures/ocr/a.png")


def test_the_container_runs_tesseract_on_one_thread():
    # OpenMP worker stacks and malloc arenas count against AP_SUBPROCESS_MAX_BYTES, so OCR runs single-threaded.
    assert re.search(r"^ENV OMP_THREAD_LIMIT=1$", _read("Dockerfile"), re.M)
