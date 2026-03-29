import time
import schedule
import subprocess
import os
from datetime import datetime

# Configuration
DISCOVERY_SCRIPT = "the_research_node.m1_cluster_discovery"
LOG_FILE = "logs/orchestrator_heartbeat.log"
os.makedirs("logs", exist_ok=True)

def heartbeat():
    # Diagnostic: Write to a log every hour to prove the script is alive
    with open(LOG_FILE, "a") as f:
        f.write(f"[HEARTBEAT] System active at {datetime.now()}\n")

def run_daily_discovery():
    print(f"\n[SYSTEM] {datetime.now()} - Starting Daily Research Phase...")
    # 'check=True' ensures it raises an error if the subprocess fails
    subprocess.run(["python", "-m", DISCOVERY_SCRIPT], check=True)
    print("[SUCCESS] curated_universe.json updated.")

if __name__ == "__main__":
    print("====== M1 Quant Pipeline Orchestrator Diagnostic Mode ======")
    
    # DIAGNOSTIC: Run a dry-run immediately to verify paths/logic
    run_daily_discovery()

    # Schedule: 4:05 PM EST Daily
    schedule.every().day.at("16:05").do(run_daily_discovery)
    
    # DIAGNOSTIC: Heartbeat every hour
    schedule.every().hour.do(heartbeat)

    while True:
        schedule.run_pending()
        time.sleep(30)