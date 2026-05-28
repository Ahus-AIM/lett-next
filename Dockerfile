FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=12 \
    MKL_NUM_THREADS=12 \
    OPENBLAS_NUM_THREADS=12 \
    NUMEXPR_NUM_THREADS=12 \
    PYTHONPATH=/workspace/app

WORKDIR /workspace/app

RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2.0" && \
    python -m pip install \
      "nibabel>=5.3.3" \
      "numpy>=2.4.4" \
      "scipy>=1.17.1"

COPY lett_next ./lett_next
COPY eval ./eval
COPY predict.sh ./predict.sh
COPY artifacts/checkpoint.pt /workspace/model/checkpoint.pt

RUN chmod +x /workspace/app/predict.sh && \
    mkdir -p /workspace/inputs /workspace/outputs /workspace/model

WORKDIR /workspace/app
CMD ["sh", "predict.sh"]
