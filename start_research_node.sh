#!/bin/bash

echo "====== RESEARCH NODE STARTUP ======"
echo "Time: $(date)"

if [ -z "$QUANT_DATA_DIR" ]; then
    echo "[FATAL] QUANT_DATA_DIR is not set. Export it before launching"
    echo "        (e.g. export QUANT_DATA_DIR=/mnt/e/quant_data)."
    exit 1
fi
echo "Vault: $QUANT_DATA_DIR"

while true; do
    uv run python -m the_research_node.orchestrator

    echo "[CRASH] Orchestrator exited at $(date)"
    echo "[RECOVERY] Restarting in 10 seconds..."
    sleep 10
done