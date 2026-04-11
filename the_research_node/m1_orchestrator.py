import time
import subprocess
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import schedule

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "orchestrator.log")
os.makedirs(LOG_DIR, exist_ok=True)

# Dual logging: terminal + rotating file
logger = logging.getLogger("M1Orchestrator")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console)

    fh = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)


def run_step(name, command):
    # Runs a subprocess and logs success or failure
    logger.info(f">> [{name}] Starting...")
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True
        )
        if result.stdout.strip():
            logger.info(f">> [{name}] {result.stdout.strip()[-200:]}")
        logger.info(f">> [{name}] Complete.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f">> [{name}] FAILED: {e.stderr.strip()[-300:]}")
        return False


def git_push():
    # Commits and pushes updated models to GitHub for the ASUS to pull
    logger.info(">> [GIT PUSH] Committing payload to GitHub...")
    try:
        subprocess.run(["git", "add", "the_models/"], check=True,
                       capture_output=True, text=True)
        subprocess.run(
            ["git", "commit", "-m",
             f"AUTO: M1 Payload Update {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
            check=True, capture_output=True, text=True
        )
        subprocess.run(["git", "push", "origin", "main"], check=True,
                       capture_output=True, text=True)
        logger.info(">> [GIT PUSH] Success.")
    except subprocess.CalledProcessError as e:
        if "nothing to commit" in str(e.stdout) + str(e.stderr):
            logger.info(">> [GIT PUSH] Nothing new to commit.")
        else:
            logger.warning(f">> [GIT PUSH] Failed: {e.stderr.strip()[-200:]}")


def run_daily_pipeline():
    # Runs Monday-Friday after market close
    # Updates data, recalculates clusters and allocations, pushes to GitHub
    logger.info(f"====== DAILY RESEARCH PIPELINE | {datetime.now().strftime('%Y-%m-%d %H:%M')} ======")
    start = time.time()

    # 1. Update macro CSV (SPY/VIX/IEF/HYG)
    run_step("MACRO UPDATE",
             ["python", "-m", "the_utilities.fetch_macro_data"])

    # 2. Incremental WRDS tick data collection
    run_step("WRDS COLLECTION",
             ["python", "-m", "the_research_node.wrds_training_collector"])

    # 3. Cluster discovery (DBSCAN + Johansen cointegration)
    if not run_step("CLUSTER DISCOVERY",
                    ["python", "-m", "the_research_node.m1_cluster_discovery"]):
        logger.error("Cluster discovery failed. Aborting pipeline.")
        return

    # 4. HRP portfolio allocation
    if not run_step("HRP ALLOCATOR",
                    ["python", "-m", "the_research_node.m1_portfolio_allocator"]):
        logger.error("HRP allocation failed. Aborting pipeline.")
        return

    # 5. Push updated universe and allocations to GitHub
    git_push()

    elapsed = (time.time() - start) / 60
    logger.info(f"====== DAILY PIPELINE COMPLETE ({elapsed:.1f} min) ======")


def run_weekly_ml_pipeline():
    # Runs Saturday morning — retrains XGBoost and rebuilds backtest matrix
    logger.info(f"====== WEEKLY ML PIPELINE | {datetime.now().strftime('%Y-%m-%d %H:%M')} ======")
    start = time.time()

    # 1. Run the full daily pipeline first to ensure data is fresh
    run_daily_pipeline()

    # 2. Rebuild the 5-year backtest matrix
    run_step("BACKTEST MATRIX",
             ["python", "-m", "the_utilities.build_backtest_matrix"])

    # 3. Retrain the XGBoost meta-labeler
    if not run_step("XGBOOST TRAINER",
                    ["python", "-m", "the_research_node.m1_xgboost_trainer"]):
        logger.error("XGBoost training failed. Keeping previous model version.")

    # 4. Push new model to GitHub
    git_push()

    elapsed = (time.time() - start) / 60
    logger.info(f"====== WEEKLY ML PIPELINE COMPLETE ({elapsed:.1f} min) ======")


if __name__ == "__main__":
    logger.info("====== M1 RESEARCH NODE ORCHESTRATOR ======")
    logger.info(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Daily pipeline: Monday-Friday @ 16:30 EST (after market close)
    schedule.every().monday.at("16:30").do(run_daily_pipeline)
    schedule.every().tuesday.at("16:30").do(run_daily_pipeline)
    schedule.every().wednesday.at("16:30").do(run_daily_pipeline)
    schedule.every().thursday.at("16:30").do(run_daily_pipeline)
    schedule.every().friday.at("16:30").do(run_daily_pipeline)

    # Weekly ML pipeline: Saturday @ 02:00 EST
    schedule.every().saturday.at("02:00").do(run_weekly_ml_pipeline)

    # Heartbeat every hour
    schedule.every().hour.do(
        lambda: logger.info(f"[HEARTBEAT] Orchestrator alive at {datetime.now()}")
    )

    logger.info("Scheduled: Daily pipeline Mon-Fri @ 16:30 EST")
    logger.info("Scheduled: Weekly ML pipeline Sat @ 02:00 EST")

    # Auto-trigger if launched after market close on a weekday
    now = datetime.now()
    if now.weekday() < 5 and now.hour >= 16:
        logger.info("Launched after market close — running daily pipeline now.")
        run_daily_pipeline()

    logger.info("Waiting for next scheduled task...\n")

    while True:
        schedule.run_pending()
        time.sleep(30)