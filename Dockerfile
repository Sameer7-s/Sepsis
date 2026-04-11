FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# Use server/app.py as the proper entry point to avoid duplicate server instances
CMD ["python", "server/app.py"]