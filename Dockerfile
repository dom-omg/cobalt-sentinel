FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy repo (excludes .git, results/, etc. via .dockerignore)
COPY . .

# Offline mode: prevents any attempt to download sentence-transformer weights.
# The MockEmbedder (hash-based) is used automatically when the real model
# is unavailable — results are reproducible and match the paper's numbers.
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Default: run all experiments and tests
CMD ["make", "all"]
