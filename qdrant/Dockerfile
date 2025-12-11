FROM qdrant/qdrant:latest

# Install minimal tooling for health checks
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*
