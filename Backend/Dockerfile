#slim Python image to keep size down
FROM python:3.11-slim

# env var
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

#ca-certificates for SSL setup
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    gnupg \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

#environment variables for SSL certificates
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

#work directory
WORKDIR /app

#Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

#port to expose
EXPOSE 8000

# Install Doppler CLI for secrets management in production
RUN curl -Ls https://cli.doppler.com/install.sh | sh

#command to run the application
# Local dev:          python main.py
# With Doppler:       doppler run -- python main.py
CMD ["python", "main.py"]