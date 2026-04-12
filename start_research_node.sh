#!/bin/bash
set -e

echo "====== RESEARCH NODE STARTUP ======"
echo "Time: $(date)"

source venv/bin/activate

while true; do
    python -m the_research_node.m1_orchestrator

    echo "[CRASH] Orchestrator exited at $(date)"
    echo "[RECOVERY] Restarting in 10 seconds..."
    sleep 10
done