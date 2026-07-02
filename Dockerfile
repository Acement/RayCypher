FROM nvidia/cuda:12.1.1-base-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python-is-python3 \
    cuda-nvrtc-12-1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/pip3 /usr/bin/pip

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir \
    ray==2.35.0 \
    numpy==1.26.4 \
    cupy-cuda12x \
    codecarbon==2.4.2 \
    psutil==5.9.8 \
    pynvml==11.5.0 \
    matplotlib==3.8.4

COPY cypher_ray.py /app/cypher_ray.py
COPY text.txt /app/text.txt

RUN mkdir -p /app/output

CMD ["python", "-m", "cProfile", "-o", "/app/output/profile.prof", "cypher_ray.py"]