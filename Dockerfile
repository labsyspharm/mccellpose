FROM python:3.12.10

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir .

ENTRYPOINT ["/bin/bash"]
