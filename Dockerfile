# CPU-only image — no CUDA, no RAPIDS, no TA-Lib
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

# Git is needed for the M1 payload sync (git pull)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install only execution node dependencies
COPY requirements.execution.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.execution.txt

# Copy the full repo (git pull needs the .git history)
COPY . .

# Create directory for runtime logs
RUN mkdir -p /app/logs

# Launch the 24/7 daemon directly
CMD ["python", "-m", "the_execution_node.main_execution"]
