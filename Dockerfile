FROM python:3.12-slim

WORKDIR /app

ENV VIRTUAL_ENV=/venv

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV OLLAMA_LOG_LEVEL=ERROR

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy project
COPY . .

# Run command 
CMD ["python", "run.py"]