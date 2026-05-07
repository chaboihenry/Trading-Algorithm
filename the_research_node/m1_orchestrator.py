import time
import subprocess
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import schedule
import sys

from the_utilities.paths import LOGS_DIR, ORCHESTRATOR_LOG, MODELS_DIR, RAW_MACRO_CSV

os.makedirs(LOGS_DIR, exist_ok=True)

logger = logging.getLogger("M1Orchestrator")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console)

    fh = RotatingFileHandler(ORCHESTRATOR_LOG, maxBytes=5*1024*1024, backupCount=5)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)


def run_step(name, command):
    # Runs a subprocess and logs success or failure
    logger.info(f"[{name}] Starting...")
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True
        )
        if result.stdout.strip():
            logger.info(f"[{name}] {result.stdout.strip()[-200:]}")
        logger.info(f"[{name}] Complete.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[{name}] FAILED: {e.stderr.strip()[-300:]}")
        return False

def git_push():
    # Commits and pushes updated models/macro to GitHub for the EC2 to pull
    # Each step is independent — one failure doesn't block subsequent steps
    logger.info("[GIT PUSH] Committing payload to GitHub...")

    # Step 1: Stage files
    try:
        subprocess.run(
            ["git", "add", f"{MODELS_DIR}/", RAW_MACRO_CSV],
            check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        logger.warning(f"[GIT PUSH] Stage failed: {e.stderr.strip()[-200:]}")
        return

    # Step 2: Commit (non-fatal if nothing to commit)
    commit_result = subprocess.run(
        ["git", "commit", "-m", f"AUTO: M1 Payload Update {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
        capture_output=True, text=True
    )
    if commit_result.returncode != 0:
        combined = (commit_result.stdout + commit_result.stderr).lower()
        if "nothing to commit" in combined or "no changes added" in combined:
            logger.info("[GIT PUSH] Nothing new to commit.")
        else:
            logger.warning(f"[GIT PUSH] Commit failed: {commit_result.stderr.strip()[-200:]}")
            return
        # Even if nothing to commit locally, still pull+push in case remote has stale refs

    # Step 3: Pull remote changes (always run — brings ASUS/EC2 pushes into sync)
    pull_result = subprocess.run(
        ["git", "pull", "--rebase", "origin", "main"],
        capture_output=True, text=True
    )
    if pull_result.returncode != 0:
        logger.warning(f"[GIT PUSH] Pull-rebase failed: {pull_result.stderr.strip()[-200:]}")
        return

    # Step 4: Push (only if local commits to push)
    push_result = subprocess.run(
        ["git", "push", "origin", "main"],
        capture_output=True, text=True
    )
    if push_result.returncode == 0:
        logger.info("[GIT PUSH] Success.")
    else:
        combined = (push_result.stdout + push_result.stderr).lower()
        if "everything up-to-date" in combined:
            logger.info("[GIT PUSH] Nothing to push — already in sync.")
        else:
            logger.warning(f"[GIT PUSH] Push failed: {push_result.stderr.strip()[-200:]}")

def run_daily_pipeline():
    # Lightweight — macro CSV update only (~30 seconds)
    logger.info(f"====== DAILY MACRO UPDATE | {datetime.now().strftime('%Y-%m-%d %H:%M')} ======")
    run_step("MACRO UPDATE",
             [sys.executable, "-m", "the_utilities.fetch_macro_data"])
    git_push()
    logger.info("====== DAILY UPDATE COMPLETE ======")

def run_research_pipeline():
    # Medium — refreshes clusters and allocations (~30-45 min)
    logger.info(f"====== RESEARCH REFRESH | {datetime.now().strftime('%Y-%m-%d %H:%M')} ======")
    start = time.time()

    run_step("MACRO UPDATE",
             [sys.executable, "-m", "the_utilities.fetch_macro_data"])

    if not run_step("CLUSTER DISCOVERY",
                    [sys.executable, "-m", "the_research_node.m1_cluster_discovery"]):
        logger.error("Cluster discovery failed. Aborting.")
        return

    if not run_step("HRP ALLOCATOR",
                    [sys.executable, "-m", "the_research_node.m1_portfolio_allocator"]):
        logger.error("HRP allocation failed. Aborting.")
        return

    git_push()

    elapsed = (time.time() - start) / 60
    logger.info(f"====== RESEARCH REFRESH COMPLETE ({elapsed:.1f} min) ======")

def run_weekly_ml_pipeline():
    # Heavy — full retrain + backtest matrix (~1-3 hours)
    logger.info(f"====== WEEKLY ML PIPELINE | {datetime.now().strftime('%Y-%m-%d %H:%M')} ======")
    start = time.time()

    run_step("MACRO UPDATE",
             [sys.executable, "-m", "the_utilities.fetch_macro_data"])

    run_step("WRDS COLLECTION",
             [sys.executable, "-m", "the_research_node.wrds_training_collector"])

    if not run_step("CLUSTER DISCOVERY",
                    [sys.executable, "-m", "the_research_node.m1_cluster_discovery"]):
        logger.error("Cluster discovery failed. Aborting.")
        return

    if not run_step("HRP ALLOCATOR",
                    [sys.executable, "-m", "the_research_node.m1_portfolio_allocator"]):
        logger.error("HRP allocation failed. Aborting.")
        return

    run_step("BACKTEST MATRIX",
             [sys.executable, "-m", "the_utilities.build_backtest_matrix"])

    if not run_step("XGBOOST TRAINER",
                    [sys.executable, "-m", "the_research_node.m1_xgboost_trainer"]):
        logger.error("XGBoost training failed. Keeping previous model.")

    git_push()

    elapsed = (time.time() - start) / 60
    logger.info(f"====== WEEKLY ML PIPELINE COMPLETE ({elapsed:.1f} min) ======")


if __name__ == "__main__":
    logger.info("====== M1 RESEARCH NODE ORCHESTRATOR ======")
    logger.info(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Monday and Thursday: cluster + HRP refresh
    schedule.every().monday.at("16:30").do(run_research_pipeline)
    schedule.every().thursday.at("16:30").do(run_research_pipeline)

    # Tuesday, Wednesday, Friday: macro CSV only
    schedule.every().tuesday.at("16:30").do(run_daily_pipeline)
    schedule.every().wednesday.at("16:30").do(run_daily_pipeline)
    schedule.every().friday.at("16:30").do(run_daily_pipeline)

    # Saturday: full ML retrain
    schedule.every().saturday.at("02:00").do(run_weekly_ml_pipeline)

    # Heartbeat every hour
    schedule.every().hour.do(
        lambda: logger.info(f"[HEARTBEAT] Orchestrator alive at {datetime.now()}")
    )

    logger.info("Scheduled: Research refresh Mon/Thu @ 16:30 EST")
    logger.info("Scheduled: Macro update Tue/Wed/Fri @ 16:30 EST")
    logger.info("Scheduled: Weekly ML pipeline Sat @ 02:00 EST")

    # Auto-trigger if launched after market close on a weekday
    now = datetime.now()
    if now.weekday() < 5 and now.hour >= 16:
        logger.info("Launched after market close — running research pipeline now.")
        run_research_pipeline()

    logger.info("Waiting for next scheduled task...\n")

    while True:
        schedule.run_pending()
        time.sleep(30)