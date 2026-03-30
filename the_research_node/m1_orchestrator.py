# m1_orchestrator.py
import time
import schedule
import subprocess
import os
from datetime import datetime

LOG_FILE = "logs/orchestrator_heartbeat.log"
os.makedirs("logs", exist_ok=True)

def heartbeat():
    with open(LOG_FILE, "a") as f:
        f.write(f"[HEARTBEAT] M1 Quant Server active at {datetime.now()}\n")

def run_daily_pipeline():
    # Runs Monday-Friday immediately after market close to prep tomorrow's execution
    print(f"\n[SYSTEM] {datetime.now()} - Initiating Daily Research Pipeline...")
    
    try:
        print("\n>> PHASE 1: Synchronizing Vault Data (Historical Backfiller)...")
        subprocess.run(["python", "-m", "the_research_node.m1_historical_backfiller"], check=True)
        
        print("\n>> PHASE 2: Engineering Structural Spreads (Cluster Discovery)...")
        subprocess.run(["python", "-m", "the_research_node.m1_cluster_discovery"], check=True)
        
        print("\n>> PHASE 3: Calculating HRP Weights (Portfolio Allocator)...")
        subprocess.run(["python", "-m", "the_research_node.m1_portfolio_allocator"], check=True)
        
        print("\n>> PHASE 4: Initiating Git Handoff to ASUS Execution Node...")
        subprocess.run(["bash", "the_research_node/sync_to_asus.sh"], check=True)
        
        print(f"\n[SUCCESS] Daily Pipeline completed at {datetime.now()}.")
    except subprocess.CalledProcessError as e:
        print(f"\n[CRITICAL ERROR] Daily Pipeline failed: {e}")

def run_weekly_ml_pipeline():
    # Runs early Saturday morning to allow massive compute time without interrupting markets
    print(f"\n[SYSTEM] {datetime.now()} - Initiating Weekly Machine Learning Pipeline...")
    
    try:
        print("\n>> PHASE 1: Training XGBoost Meta-Labeler...")
        subprocess.run(["python", "-m", "the_research_node.m1_xgboost_trainer"], check=True)
        
        print("\n>> PHASE 2: Initiating Git Handoff to ASUS Execution Node...")
        subprocess.run(["bash", "the_research_node/sync_to_asus.sh"], check=True)
        
        print(f"\n[SUCCESS] Weekly ML Pipeline completed at {datetime.now()}.")
    except subprocess.CalledProcessError as e:
        print(f"\n[CRITICAL ERROR] Weekly ML Pipeline failed: {e}")

if __name__ == "__main__":
    print("====== M1 Quant Pipeline Orchestrator Active ======")

    # Schedule Daily Pipeline (16:05 EST - Market Close)
    # Note: Running every day is safe; the backfiller simply skips if markets were closed.
    schedule.every().day.at("16:05").do(run_daily_pipeline)
    
    # Schedule Weekly ML Pipeline (Saturday at 02:00 AM)
    schedule.every().saturday.at("02:00").do(run_weekly_ml_pipeline)
    
    # Heartbeat every hour
    schedule.every().hour.do(heartbeat)

    while True:
        schedule.run_pending()
        time.sleep(30)