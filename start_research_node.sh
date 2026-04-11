#!/bin/bash
set -e

echo "====== RESEARCH NODE STARTUP ======"
echo "Time: $(date)"

# Activate virtual environment
source venv/bin/activate

# Launch orchestrator with auto-restart
while true; do
    python -m the_research_node.m1_orchestrator

    EXIT_CODE=$?
    echo "[CRASH] Orchestrator exited with code $EXIT_CODE at $(date)"
    echo "[RECOVERY] Restarting in 10 seconds..."
    sleep 10
done