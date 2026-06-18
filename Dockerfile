# CPU-only image — no CUDA, no RAPIDS, no TA-Lib
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

# Git is needed for the M1 payload sync (git pull)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install only execution node dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the full repo (git pull needs the .git history)
COPY . .

# Create directory for runtime logs
RUN mkdir -p /app/logs

# Configure git so EOD `git pull` from research node succeeds inside the container
RUN git config --global --add safe.directory /app && \
    git config --global user.email "bot@trading-algorithm.local" && \
    git config --global user.name "Execution Node Bot"

# Launch the 24/7 daemon directly
CMD ["python", "-m", "the_execution_node.main_execution"]
