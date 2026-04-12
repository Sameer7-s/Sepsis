FROM python:3.10-slim

# Hugging Face Spaces requires a non-root user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Install dependencies first (layer-cached unless requirements.txt changes)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the codebase
COPY --chown=user . .

# HF Spaces expects the app on port 7860
EXPOSE 7860
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Single entrypoint — server/app.py starts uvicorn once and only once
ENTRYPOINT ["python", "-u", "server/app.py"]