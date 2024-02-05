FROM python:3.11-slim

WORKDIR /workspace

RUN python -m pip install --upgrade pip

COPY requirements.txt /workspace/
COPY bird_predictor.py /workspace/

RUN pip install -r requirements.txt