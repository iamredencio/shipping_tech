# Use a stable base image
FROM python:3.9-bullseye

# Set working directory
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update
RUN apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy dependency file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code into the container
COPY app/ app/
COPY scripts/ scripts/
COPY tests/ tests/

# Optional: Add data directory if required
COPY data/ data/

# Set environment variables
ENV DATA_PATH=/app/data

# Expose the port your app runs on
EXPOSE 8001

# Run the FastAPI application
CMD ["uvicorn", "app.api.endpoints:app", "--host", "0.0.0.0", "--port", "8001"]
