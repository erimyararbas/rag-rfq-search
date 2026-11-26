FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /app
# CPU-only torch first (same version uv.lock pins, but from the CPU index; requirements.lock.txt
# excludes torch and the CUDA wheels), so sentence-transformers does not pull CUDA builds
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml requirements.lock.txt ./
RUN pip install -r requirements.lock.txt
COPY app/ app/
COPY scripts/ scripts/
COPY tests/ tests/
COPY data/ data/
COPY eval/ eval/
EXPOSE 8000
