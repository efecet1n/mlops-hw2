# Dockerfile for Flight Delay Prediction Service
# MLOps HW2 - Efe Çetin

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Copy model (if exists)
COPY model/ ./model/

# Expose port
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV PYTHONPATH=/app

# Run the API
CMD ["python", "-m", "src.api"]
