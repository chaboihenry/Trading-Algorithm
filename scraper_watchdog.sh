#!/bin/bash

# scraper_watchdog.sh
# Ensure the script stops if it encounters a severe OS-level error
set -e 

echo "====== M1 Scraper Watchdog Initialized ======"

# Infinite loop to keep the scraper alive
while true; do
    echo "[SYSTEM] Launching the_research_node.m1_scraper..."
    
    # Run the Python scraper
    python -m the_research_node.m1_scraper
    
    # If the Python script crashes or exits, the loop continues to this line
    echo "[CRASH DETECTED] Scraper process terminated abruptly."
    echo "[RECOVERY] Restarting stream in 5 seconds..."
    
    sleep 5
done