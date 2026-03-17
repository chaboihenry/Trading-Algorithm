# Use NVIDIA RAPIDS base image (Includes cuDF, cuML, GPU XGBoost for RTX 5080)
FROM rapidsai/base:24.10-cuda12.5-py3.11

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app
USER root

# PEPE: Added 'git' to the apt-get install list to enable version control
RUN apt-get update && apt-get install -y --no-install-recommends build-essential wget git \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib && ./configure --prefix=/usr && make && make install \
    && cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Create directory for runtime logs
RUN mkdir -p /app/logs

CMD ["/bin/bash"]