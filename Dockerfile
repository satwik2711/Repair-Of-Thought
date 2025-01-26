FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p outputs/sol outputs/patches outputs/val datasets

EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "ui.py"]