# Hospital Bed Allocation — OpenEnv v1
# Hugging Face Spaces (Docker SDK) — port 7860
#
# Build:  docker build -t hospital-bed-env .
# Run:    docker run -p 7860:7860 \
#           -e API_BASE_URL=https://api.openai.com/v1 \
#           -e MODEL_NAME=gpt-4o-mini \
#           -e HF_TOKEN=hf_xxx \
#           hospital-bed-env

FROM python:3.11-slim

LABEL maintainer="openenv-submission"
LABEL env_id="HospitalBedAllocation-v1"
LABEL version="1.0.0"

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY environment.py graders.py baseline.py inference.py app.py openenv.yaml README.md ./

# Non-root user (required by HF Spaces)
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
ENV PORT=7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

HEALTHCHECK --interval=20s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
