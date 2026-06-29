FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir \
    ray==2.35.0 \
    numpy==1.26.4 \
    codecarbon==2.4.2 \
    psutil==5.9.8 \
    pynvml==11.5.0

COPY cypher_ray.py /app/cypher_ray.py

COPY text.txt /app/text.txt

CMD ["python", "-m", "cProfile", "-o", "/app/output/profile.prof", "cypher_ray.py"]