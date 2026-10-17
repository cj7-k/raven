# One image, used by both the backend and the frontend (docker-compose.yml
# picks which one to start). Python 3.12 on purpose: some pinned packages
# (numpy 1.26, chroma-hnswlib) have prebuilt wheels for 3.12 but not 3.13,
# and a slim image has no compiler to build them from source.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=10

WORKDIR /app

# Install dependencies first, so this (slow) layer is cached and only
# re-runs when requirements-docker.txt changes - not on every code edit.
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# Application code. Data (chunks, vector store) and the embedding model are
# NOT baked in - docker-compose.yml mounts them.
COPY src/ ./src/
COPY frontend/ ./frontend/

EXPOSE 8000 8501

# Default: run the GraphQL backend. The frontend service overrides this.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
