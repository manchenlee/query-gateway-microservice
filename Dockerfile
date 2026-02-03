FROM golang:1.24-alpine AS k6-builder
RUN apk add --no-cache git
RUN go install go.k6.io/xk6/cmd/xk6@v0.11.0
RUN xk6 build --with github.com/grafana/xk6-dashboard@latest --output /k6

FROM python:3.11-slim
WORKDIR /app

ENV TRANSFORMERS_CACHE=/app/model_cache
ENV MODEL_NAME="intfloat/e5-small-v2"

COPY --from=k6-builder /k6 /usr/local/bin/k6

RUN sed -i 's/deb.debian.org/ftp.tw.debian.org/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update || true && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

ENV HF_ENDPOINT=https://hf-mirror.com
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}')"

COPY . .

RUN chmod +x run_test.sh

EXPOSE 8000 5665

ENTRYPOINT ["./run_test.sh"]