FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1 git \
    && rm -rf /var/lib/apt/lists/*

ENV NUMBA_CACHE_DIR=/tmp
ENV CELLPOSE_LOCAL_MODELS_PATH=/opt/cellpose

COPY / /app
# 1. Install the app itself.
# 2. Download the needed model weights.
RUN pip install --no-cache-dir /app \
    && mkdir -p $CELLPOSE_LOCAL_MODELS_PATH \
    && python -c "import cellpose.models; cellpose.models.CellposeModel()" \
    && chmod -R a+rw /tmp $CELLPOSE_LOCAL_MODELS_PATH
