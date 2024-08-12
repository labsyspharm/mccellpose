FROM biocontainers/cellpose:3.0.1_cv1

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U \
    cellpose

RUN pip install --no-cache-dir \
    dask \
    scipy \
    numpy \
    dask-image \
    palom

COPY /cellpose-large-img.py /app/

ENTRYPOINT ["/bin/bash"]
