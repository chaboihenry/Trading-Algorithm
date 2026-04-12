#!/bin/bash
set -e

echo "====== EXECUTION NODE STARTUP ======"
echo "Time: $(date)"

git pull origin main

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
    python -m the_execution_node.main_execution

    echo "[CRASH] Daemon exited at $(date)"
    echo "[RECOVERY] Restarting in 30 seconds..."
    sleep 30
done