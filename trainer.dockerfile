# Base image
FROM python:3.8-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

COPY reports/ reports/
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/


ENTRYPOINT ["python", "-u", "src/models/train_model.py","train"]


