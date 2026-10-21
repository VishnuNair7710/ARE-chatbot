# syntax=docker/dockerfile:1
# Two runtime images from one file:  docker build --target api .   |   docker build --target ui .
# No secrets are baked in: pass LLM_API_KEY etc. at run time (env or compose env_file).

FROM python:3.11-slim AS deps-api
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
COPY requirements.txt /tmp/requirements.txt
RUN /opt/venv/bin/pip install -r /tmp/requirements.txt

FROM python:3.11-slim AS deps-ui
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
COPY requirements-ui.txt /tmp/requirements-ui.txt
RUN /opt/venv/bin/pip install -r /tmp/requirements-ui.txt

# ---- API ---------------------------------------------------------------------------------------
FROM python:3.11-slim AS api
ENV PATH="/opt/venv/bin:$PATH" PYTHONPATH=/app/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    CHROMA_DIR=/data/chroma CHECKPOINT_DB=/data/sessions.db LEAD_DB_PATH=/data/leads.db \
    TURN_LOG_PATH=/data/turns.jsonl FASTEMBED_CACHE_PATH=/data/models
RUN useradd --create-home --uid 10001 app && mkdir /data && chown app /data
COPY --from=deps-api /opt/venv /opt/venv
WORKDIR /app
COPY src ./src
COPY data ./data
USER app
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "arbot.services.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- UI ------------------------------------------------------------------------------------------
FROM python:3.11-slim AS ui
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 API_URL=http://api:8000
RUN useradd --create-home --uid 10001 app
COPY --from=deps-ui /opt/venv /opt/venv
WORKDIR /app
COPY ui ./ui
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health', timeout=4)"
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
