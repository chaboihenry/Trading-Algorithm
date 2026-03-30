#!/bin/bash

# orchestrator_watchdog.sh
set -e

echo "====== M1 Orchestrator Watchdog Initialized ======"

while true; do
    echo "[SYSTEM] Launching m1_orchestrator.py..."
    
    python -m the_research_node.m1_orchestrator
    
    echo "[CRASH DETECTED] Orchestrator process terminated."
    echo "[RECOVERY] Restarting scheduler in 5 seconds..."
    
    sleep 5
done
