import os
import time
import json
import threading
import subprocess
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

from the_execution_node.core.live_streamer import LiveStreamer
from the_execution_node.core.order_router import OrderRouter
from the_execution_node.strategies.stat_arb_engine import generate_signals, check_exits
from the_utilities.strategy_config import (
    COOLDOWN_MINUTES, EOD_LIQUIDATION_TIME_MINUTES, SAFE_ENTRY_WINDOW,
    EOD_COOLDOWN_SKIP_MINUTES
)

class ExecutionOrchestrator:
    MIN_BUFFER = 50
    EVAL_INTERVAL = 60

    def __init__(self):
        self._setup_logging()
        self.logger.info("Initializing Execution Orchestrator...")

        load_dotenv()
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_API_SECRET")
        self.base_url = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")

        self.models_dir = "the_models"
        self.router = OrderRouter(self.api_key, self.secret_key, self.base_url)

        # Position tracking: spread_name -> position metadata from stat_arb_engine
        self.open_positions = {}

        # Cooldwown tracking: spread_name -> cooldown expiry timestamp
        self.cooldown_tracker = {}

        self.streamer = None

    def _setup_logging(self):
        # Dual logging: terminal output + rotating file output
        os.makedirs("logs", exist_ok=True)
        log_file = "logs/execution_node.log"

        self.logger = logging.getLogger("QuantNode")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            # Terminal
            console_handler = logging.StreamHandler()
            console_format = logging.Formatter('%(levelname)s: %(message)s')
            console_handler.setFormatter(console_format)

            # File (max 5MB, keeps last 5 backups)
            file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
            file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_format)

            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

    def _sync_m1_payload(self):
        # Pull the latest research models and universe from GitHub
        self.logger.info("Synchronizing with M1 Research Node (Git Pull)...")
        try:
            result = subprocess.run(
                ["git", "pull", "origin", "main"],
                capture_output=True, text=True, check=True
            )
            self.logger.info(f"GitHub Sync: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Git Pull failed: {e.stderr.strip()}")

    def _save_position_state(self):
        # Persist open positions to disk so we survive restarts
        state_path = os.path.join("logs", "open_positions.json")
        try:
            with open(state_path, "w") as f:
                json.dump(self.open_positions, f, indent=4, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save position state: {e}")

    def _load_position_state(self):
        # Restore open positions from disk after a restart
        state_path = os.path.join("logs", "open_positions.json")
        if os.path.exists(state_path):
            try:
                with open(state_path, "r") as f:
                    self.open_positions = json.load(f)
                self.logger.info(f"Restored {len(self.open_positions)} open positions from disk.")
            except Exception as e:
                self.logger.warning(f"Failed to load position state: {e}")
                self.open_positions = {}

    def _is_safe_time(self):
        now = datetime.now()
        current_minutes = now.hour * 60 + now.minute
        start_min, end_min = SAFE_ENTRY_WINDOW
        return start_min <= current_minutes <= end_min
    
    def _force_eod_liquidation(self):
        now = datetime.now()
        current_minutes = now.hour * 60 + now.minute
        if current_minutes < EOD_LIQUIDATION_TIME_MINUTES:
            return

        self.logger.info("[EOD] 15:50 ET — Force-liquidating all positions to avoid overnight risk.")

        for spread_name in list(self.open_positions.keys()):
            pos_data = self.open_positions.get(spread_name, {})
            weights = pos_data.get('johansen_weights', {})

            if self.router.close_spread(spread_name, weights, "eod_liquidation"):
                del self.open_positions[spread_name]
                self.cooldown_tracker[spread_name] = datetime.now()
                self._save_position_state()
                self.logger.info(f"[EOD] {spread_name} liquidated.")
            else:
                self.logger.warning(f"[EOD] Failed to liquidate {spread_name}.")

    def _process_exits(self, matrix: pd.DataFrame):
        # Check all open positions for exit conditions
        if not self.open_positions:
            return

        exits = check_exits(matrix, self.open_positions, self.models_dir)

        for spread_name, reason in exits.items():
            pos_data = self.open_positions.get(spread_name, {})
            weights = pos_data.get('johansen_weights', {})

            self.logger.info(f"[EXIT] {spread_name} | Reason: {reason}")

            if self.router.close_spread(spread_name, weights, reason):
                del self.open_positions[spread_name]
                self.cooldown_tracker[spread_name] = datetime.now()
                self._save_position_state()
                self.logger.info(f"[CLOSED] {spread_name} successfully. Cooldown: {self.COOLDOWN_MINUTES}min")
            elif reason in ("basket_removed", "missing_legs", "invalid_position"):
                # Nothing to close in Alpaca — purge the phantom from state
                del self.open_positions[spread_name]
                self._save_position_state()
                self.logger.info(f"[PURGED] {spread_name} removed from state (not held in Alpaca).")
            else:
                self.logger.warning(f"[WARNING] Failed to close {spread_name}.")

    def _process_entries(self, matrix: pd.DataFrame):
        # Generate new signals and route approved trades
        if not self._is_safe_time():
            return

        signals, _ = generate_signals(matrix, self.models_dir)

        if not signals:
            return

        open_spread_names = set(self.open_positions.keys())

        for spread_name, signal_data in signals.items():
            # Skip if already holding this spread
            if spread_name in open_spread_names:
                continue
            
            # Skip if the spread is on cooldown
            last_exit = self.cooldown_tracker.get(spread_name)
            if last_exit and (datetime.now() - last_exit) < timedelta(minutes=COOLDOWN_MINUTES):
            continue

            target_pos = signal_data['target_position']
            z_score = signal_data['current_z']
            ai_conf = signal_data['ai_confidence']
            bet_size = signal_data['bet_size']
            action = 'LONG' if target_pos == 1 else 'SHORT'

            self.logger.info(
                f"[SIGNAL] {spread_name} | {action} | "
                f"Z={z_score:.2f} | AI={ai_conf:.3f} | Bet={bet_size:.3f}"
            )

            # Route through order router
            success = self.router.execute_spread(
                spread_name, signal_data, matrix, open_spread_names
            )

            if success:
                # Track the position with all metadata for exit evaluation
                self.open_positions[spread_name] = signal_data
                self.open_positions[spread_name]['bars_held'] = 0
                self.open_positions[spread_name]['entry_time'] = datetime.now().isoformat()
                self._save_position_state()

                self.logger.info(f"[ENTERED] {spread_name} | {action}")

    def _increment_bars_held(self):
        # Tick the bar counter for all open positions each evaluation cycle
        for spread_name in self.open_positions:
            self.open_positions[spread_name]['bars_held'] = (
                self.open_positions[spread_name].get('bars_held', 0) + 1
            )

    def _execute_trading_day(self):
        # The intra-day execution loop. Runs ONLY during market hours.
        self.logger.info("====== MARKET OPEN: INITIATING TRADING SEQUENCE ======")

        # Load any positions that survived a restart
        self._load_position_state()

        # Start the live data stream
        self.streamer = LiveStreamer(
            self.api_key, self.secret_key, self.base_url, self.models_dir
        )
        stream_thread = threading.Thread(target=self.streamer.start_streaming, daemon=True)
        stream_thread.start()

        self.logger.info(f"Warming up streamer buffer ({self.MIN_BUFFER} bars needed)...")
        time.sleep(self.EVAL_INTERVAL)

        while self.router.api.get_clock().is_open:
            try:
                matrix = self.streamer.get_latest_matrix()

                if matrix.empty or len(matrix) < self.MIN_BUFFER:
                    self.logger.info(
                        f"[BUFFERING] {len(matrix)}/{self.MIN_BUFFER} bars collected..."
                    )
                    time.sleep(self.EVAL_INTERVAL)
                    continue

                self.logger.info(
                    f"[EVAL] Matrix: {matrix.shape} | "
                    f"Open Positions: {len(self.open_positions)} | "
                    f"Time: {datetime.now().strftime('%H:%M:%S')}"
                )

                # 1. Check exits first — free up capital before entering new trades
                self._process_exits(matrix)

                # 2. Force-close everything near market close
                self._force_eod_liquidation()

                # 3. Look for new entry signals
                self._process_entries(matrix)

                # 4. Increment bar counters for time barrier tracking
                self._increment_bars_held()

                # 5. Persist state after every cycle
                self._save_position_state()

            except Exception as e:
                self.logger.error(f"Evaluation cycle error: {e}", exc_info=True)

            time.sleep(self.EVAL_INTERVAL)

        # Market closed — clean up
        self.logger.info("Market closed. Terminating WebSocket.")
        if self.streamer and self.streamer.stream:
            try:
                self.streamer.stream.stop()
            except Exception:
                pass

        self._save_position_state()
        self.logger.info(
            f"End of day. {len(self.open_positions)} positions carried overnight."
        )

    def run_24_7_daemon(self):
        # The master infinite loop that manages sleep cycles
        self.logger.info("====== BOOTING QUANT 24/7 DAEMON ======")

        # Log account status on boot
        try:
            equity, bp = self.router.get_account_metrics()
            self.logger.info(f"Account Equity: ${equity:.2f} | Buying Power: ${bp:.2f}")
        except Exception as e:
            self.logger.error(f"Could not fetch account metrics: {e}")

        while True:
            try:
                clock = self.router.api.get_clock()

                if clock.is_open:
                    # Sync latest models before trading
                    self._sync_m1_payload()
                    self._execute_trading_day()
                else:
                    # Market closed — sync and hibernate
                    self._sync_m1_payload()

                    clock = self.router.api.get_clock()
                    time_to_open = (clock.next_open - clock.timestamp).total_seconds()
                    hours = time_to_open / 3600

                    self.logger.info(f"Market CLOSED. Next open in {hours:.2f} hours.")
                    self.logger.info("Hibernating until market open...")

                    time.sleep(time_to_open)
                    self.logger.info("ALARM: Market open detected. Waking up...")

            except KeyboardInterrupt:
                self.logger.info("Daemon terminated by user.")
                self._save_position_state()
                break

            except Exception as e:
                self.logger.error(f"DAEMON ERROR: {e}", exc_info=True)
                self.logger.info("Recovering in 60 seconds...")
                time.sleep(60)


if __name__ == "__main__":
    orchestrator = ExecutionOrchestrator()
    try:
        orchestrator.run_24_7_daemon()
    except KeyboardInterrupt:
        orchestrator.logger.info("24/7 Daemon terminated by KeyboardInterrupt.")
        orchestrator._save_position_state()