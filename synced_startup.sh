#!/bin/bash

echo "====== EXECUTION NODE STARTUP SEQUENCE ======"

# 1. Sync the latest M1 Payload from GitHub
echo "[SYSTEM] Pulling latest strategy payload from GitHub..."
git pull origin main

# 2. Verify the payload exists
if [ ! -f "the_models/curated_universe.json" ]; then
    echo "[CRITICAL] curated_universe.json not found! Halting startup."
    exit 1
fi

echo "[SUCCESS] Payload verified. Launching Master Loop..."

# 3. Launch the Orchestrator
python -m the_execution_node.main_execution