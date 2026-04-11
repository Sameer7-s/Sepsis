FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# Set environment variables for proper cleanup and signal handling
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Use exec form to ensure proper signal handling (PID 1 for proper cleanup)
ENTRYPOINT ["python", "-u", "server/app.py"]