# Hospital Bed Allocation — OpenEnv v1
# Deploys as a Hugging Face Space (Gradio app on port 7860)
# Build: docker build -t hospital-bed-env .
# Run:   docker run -p 7860:7860 hospital-bed-env

FROM python:3.11-slim

LABEL maintainer="openenv-submission"
LABEL env_id="HospitalBedAllocation-v1"
LABEL version="1.0.0"

# ── System deps ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ─────────────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source ─────────────────────────────────────────────────────────────
COPY environment.py graders.py baseline.py openenv.yaml app.py ./
COPY README.md .

# ── Non-root user (Hugging Face requirement) ────────────────────────────────
RUN useradd -m -u 1000 appuser
USER appuser

# ── Expose Gradio port ──────────────────────────────────────────────────────
EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

# ── Healthcheck ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "app.py"]
