#!/bin/bash

echo "==== Initating Handoff to Execution Mode ==="

# 1. Add only the specific JSON configuration files
git add curated_universe.json
git add model/meta_labeler_v1.json

# 2. Commit the new parameters
TIMESTAMP=$(date + "%Y-%m-%d_%H:%M:%S")
git commit -m "Automated Model & Universe Update: $TIMESTAMP"

# 3. Push to Github
git push origin main

echo "=== Handoff Complete. Asus Ready to Pull. ==="
