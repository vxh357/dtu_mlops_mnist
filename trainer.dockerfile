# Base image
FROM --platform=linux/amd64 python:3.10-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY dtu_mlops_mnist/ dtu_mlops_mnist/
COPY data/ data/
COPY models/ models/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -e . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "dtu_mlops_mnist/train_model.py"]
