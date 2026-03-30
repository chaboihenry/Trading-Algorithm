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
    """Runs Monday-Friday @ 16:05 EST to prepare the next day's payload."""
    print(f"\n[SYSTEM] {datetime.now()} - Initiating Daily Research Pipeline...")
    
    try:
        print(">> [1/4] Running Historical Backfiller...")
        subprocess.run(["python", "-m", "the_research_node.m1_historical_backfiller"], check=True)
        
        print(">> [2/4] Running Cluster Discovery...")
        subprocess.run(["python", "-m", "the_research_node.m1_cluster_discovery"], check=True)
        
        print(">> [3/4] Running HRP Portfolio Allocator...")
        subprocess.run(["python", "-m", "the_research_node.m1_portfolio_allocator"], check=True)
        
        print(">> [4/4] Initiating Git Handoff to ASUS...")
        subprocess.run(["bash", "the_research_node/sync_to_asus.sh"], check=True)
        
        print(f"\n[SUCCESS] Daily Pipeline completed at {datetime.now()}.")
    except subprocess.CalledProcessError as e:
        print(f"\n[CRITICAL ERROR] Daily Pipeline failed: {e}")

def run_weekly_ml_pipeline():
    """Runs Saturday @ 02:00 AM to retrain the XGBoost Meta-Labeler."""
    print(f"\n[SYSTEM] {datetime.now()} - Initiating Weekly Machine Learning Pipeline...")
    
    try:
        print("\n>> [PHASE 1] Forcing Daily Prep for Pristine Data...")
        # Step 1: Ensure data is perfect before training
        subprocess.run(["python", "-m", "the_research_node.m1_historical_backfiller"], check=True)
        subprocess.run(["python", "-m", "the_research_node.m1_cluster_discovery"], check=True)
        subprocess.run(["python", "-m", "the_research_node.m1_portfolio_allocator"], check=True)
        
        print("\n>> [PHASE 2] Training XGBoost Meta-Labeler...")
        subprocess.run(["python", "-m", "the_research_node.m1_xgboost_trainer"], check=True)
        
        print("\n>> [PHASE 3] Initiating Git Handoff to ASUS...")
        subprocess.run(["bash", "the_research_node/sync_to_asus.sh"], check=True)
        
        print(f"\n[SUCCESS] Weekly ML Pipeline completed at {datetime.now()}.")
    except subprocess.CalledProcessError as e:
        print(f"\n[CRITICAL ERROR] Weekly ML Pipeline failed: {e}")

if __name__ == "__main__":
    print("====== M1 Quant Pipeline Orchestrator Active ======")

    # 1. Schedule Daily Pipeline (Monday - Friday @ 16:05 EST)
    schedule.every().monday.at("16:05").do(run_daily_pipeline)
    schedule.every().tuesday.at("16:05").do(run_daily_pipeline)
    schedule.every().wednesday.at("16:05").do(run_daily_pipeline)
    schedule.every().thursday.at("16:05").do(run_daily_pipeline)
    schedule.every().friday.at("16:05").do(run_daily_pipeline)
    
    # 2. Schedule Weekly ML Pipeline (Saturday @ 02:00 AM)
    schedule.every().saturday.at("02:00").do(run_weekly_ml_pipeline)
    
    # Heartbeat every hour
    schedule.every().hour.do(heartbeat)

    while True:
        schedule.run_pending()
        time.sleep(30)