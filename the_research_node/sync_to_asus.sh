#!/bin/bash

echo "==== Initiating Handoff to Execution Node ==="

# 1. Target the correct 'the_models' directory
git add the_models/curated_universe.json
git add the_models/meta_labeler_v1.json

# 2. Commit the new parameters
TIMESTAMP=$(date +"%Y-%m-%d_%H:%M:%S")
git commit -m "Automated Model & Universe Update: $TIMESTAMP"

# 3. Push to Github
git push origin main

echo "=== Handoff Complete. ASUS Ready to Pull. ==="