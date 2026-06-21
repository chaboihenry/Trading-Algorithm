#!/bin/bash

echo "====== EXECUTION NODE STARTUP ======"
echo "Time: $(date)"

if [ -z "$QUANT_DATA_DIR" ]; then
    echo "[FATAL] QUANT_DATA_DIR is not set. Export it before launching"
    echo "        (e.g. export QUANT_DATA_DIR=/mnt/e/quant_data)."
    exit 1
fi
echo "Vault: $QUANT_DATA_DIR"

git pull origin main || echo "[WARN] git pull failed; using existing payload"

if [ ! -f "the_models/curated_universe.json" ]; then
    echo "[CRITICAL] curated_universe.json missing. Cannot start."
    exit 1
fi

if [ ! -f "the_models/active_model_version.txt" ]; then
    echo "[CRITICAL] active_model_version.txt missing. Cannot start."
    exit 1
fi

echo "[SUCCESS] Payload verified."

while true; do
    uv run python -m the_execution_node.main_execution

    echo "[CRASH] Daemon exited at $(date)"
    echo "[RECOVERY] Restarting in 30 seconds..."
    sleep 30
done