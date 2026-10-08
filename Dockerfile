# AP Copilot dashboard for Cloud Run: FastAPI + ADK workflow, with the reader's OCR tools installed.
# The image holds no secrets: ANTHROPIC_API_KEY is injected at runtime (Secret Manager on Cloud Run).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# poppler-utils: pdftotext and pdftoppm. Tesseract with English (pulled in by tesseract-ocr), Spanish and both Chinese models.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-spa \
        tesseract-ocr-chi-sim \
        tesseract-ocr-chi-tra \
    && rm -rf /var/lib/apt/lists/* \
    && tesseract --list-langs | grep -qx eng \
    && tesseract --list-langs | grep -qx spa

RUN useradd --system --uid 10001 --create-home --home-dir /home/app app

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# .dockerignore decides what reaches the image: no .env files, no eval/out, no tests except the offline LLM fixtures.
COPY --chown=app:app . .

# Cloud Run's filesystem is ephemeral, so the spend ledger lives in /tmp and resets on every cold start. The per-instance
# cap (AP_LLM_SPEND_CAP_USD) is therefore a per-instance guard only; see DEPLOY.md for the cross-instance stop.
ENV AP_SPEND_LEDGER_PATH=/tmp/ap-spend/spend.json

USER app

# Cloud Run sets PORT; 8080 is the fallback for a local `docker run`.
CMD ["sh", "-c", "exec uvicorn web.server:web_app --host 0.0.0.0 --port ${PORT:-8080}"]
