# Use a slim Python image to keep size down
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── FIX: DDG requires proper SSL cert chain + CA bundle ──────────────────────
# python:3.11-slim strips ca-certificates; without them DDG HTTPS fails
# silently and the retry loop burns through all attempts → rate-limit errors.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── FIX: Point Python's SSL to the system cert store ────────────────────────
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose the port your app runs on (usually 8000 for FastAPI/Flask)
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"]