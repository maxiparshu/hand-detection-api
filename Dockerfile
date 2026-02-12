FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements_docker.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -v -r requirements.txt

COPY gestures/ ./gestures/
COPY neural_network/ ./neural_network/
COPY services/ ./services/

COPY main_server.py .

RUN mkdir -p temp

EXPOSE 8000

CMD ["uvicorn", "main_server:app", "--host", "0.0.0.0", "--port", "8000"]