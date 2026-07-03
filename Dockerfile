FROM python:3.10-slim

# Install system dependencies required by MediaPipe's C++ backend
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgles2 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Render assigns a dynamic port via the PORT environment variable
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
