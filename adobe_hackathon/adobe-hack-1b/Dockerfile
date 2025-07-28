FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir sentence-transformers pdfplumber scikit-learn

COPY app.py .
COPY persona.txt .

RUN mkdir -p /app/input /app/output

CMD ["python", "app.py"]
