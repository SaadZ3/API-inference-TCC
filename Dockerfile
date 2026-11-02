# Empacota a API Python com o n?cleo local e as depend?ncias fixadas.
FROM python:3.13-slim-bookworm
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY requirements.lock.txt ./
# A imagem usa CPU, sem bibliotecas CUDA.
RUN python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install -r requirements.lock.txt
COPY vendor ./vendor
COPY pyproject.toml requirements.txt ./
COPY src ./src
# A instalação editável preserva a resolução dos caminhos relativos a /app.
RUN python -m pip install --no-deps -e ./vendor/privacy-policy-core -e . && python -m pip check
COPY model ./model
COPY config ./config
COPY scripts/start_railway.py ./scripts/start_railway.py
ENV MODEL_DIR=/app/model SCORING_RULES=/app/config/scoring.yaml
ENV RECOMMENDATIONS_CATALOG=/app/config/recommendations.yaml
ENV DEVICE=cpu BATCH_SIZE=4 CPU_THREADS=4
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PORT=8000
EXPOSE 8000
CMD ["python", "scripts/start_railway.py"]
