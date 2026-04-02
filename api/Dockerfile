FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .

# GeoLite2-City.mmdb is mounted at runtime via /data volume.
# Download from MaxMind (free account) and place in the data directory.

EXPOSE 11400

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "11400"]
