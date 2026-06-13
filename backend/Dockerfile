FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

WORKDIR /app

COPY extra_requirements.txt /app
RUN pip install -r /app/extra_requirements.txt

COPY backend/ /app/backend
COPY data/ /app/data
