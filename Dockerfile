FROM python:3.11-slim

WORKDIR /app

# 安裝系統層級依賴 (libgl 用於 opencv-python-headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安裝 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 複製專案依賴描述檔
COPY pyproject.toml uv.lock ./

# 安裝所有生產依賴 (不含 dev)
RUN uv sync --frozen --no-dev

# 複製原始碼與模型權重
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

# 啟動命令
CMD ["uv", "run", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]