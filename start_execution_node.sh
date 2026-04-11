#!/bin/bash
set -e

echo "====== EXECUTION NODE STARTUP ======"
echo "Time: $(date)"

# 1. Pull latest models from GitHub
echo "[SYSTEM] Syncing latest M1 payload..."
git pull origin main

# 2. Verify critical files exist
if [ ! -f "the_models/curated_universe.json" ]; then
    echo "[CRITICAL] curated_universe.json missing. Cannot start."
    exit 1
fi

if [ ! -f "the_models/active_model_version.txt" ]; then
    echo "[CRITICAL] active_model_version.txt missing. Cannot start."
    exit 1
fi

echo "[SUCCESS] Payload verified."

# 3. Launch the trading daemon with auto-restart on crash
echo "[SYSTEM] Launching 24/7 trading daemon..."

while true; do
    python -m the_execution_node.main_execution

    EXIT_CODE=$?
    echo "[CRASH] Daemon exited with code $EXIT_CODE at $(date)"
    echo "[RECOVERY] Restarting in 30 seconds..."
    sleep 30
done

