import os
import time
import threading
import subprocess
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
from dotenv import load_dotenv

from the_execution_node.core.live_streamer import LiveStreamer
from the_execution_node.core.order_router import OrderRouter
from the_execution_node.strategies.stat_arb_engine import generate_signals

class ExecutionOrchestrator:
    # The 24/7 Master Daemon of the Execution Node.
    def __init__(self):
        self._setup_logging()
        self.logger.info("Initializing Execution Orchestrator...")
        
        load_dotenv()
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")
        
        self.models_dir = "the_models"
        self.router = OrderRouter(self.api_key, self.secret_key, self.base_url)
        
        self.active_positions = set()
        self.streamer = None

    def _setup_logging(self):
        # Configures dual-logging: Terminal output + Rotating File output
        os.makedirs("logs", exist_ok=True)
        log_file = "logs/execution_node.log"
        
        self.logger = logging.getLogger("QuantNode")
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate logs if initialized multiple times
        if not self.logger.handlers:
            # 1. Terminal Output
            console_handler = logging.StreamHandler()
            console_format = logging.Formatter('%(levelname)s: %(message)s')
            console_handler.setFormatter(console_format)
            
            # 2. File Output (Max 5MB per file, keeps last 5 backups)
            file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
            file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_format)
            
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

    def _sync_m1_payload(self):
        # Automatically pulls the latest JSON research from GitHub
        self.logger.info("Synchronizing with M1 Research Node (Git Pull)...")
        try:
            result = subprocess.run(
                ["git", "pull", "origin", "main"], 
                capture_output=True, text=True, check=True
            )
            self.logger.info(f"GitHub Sync Complete: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Git Pull failed (Offline?). Error: {e.stderr.strip()}")

    def _execute_trading_day(self):
        # The intra-day execution loop. Runs ONLY during market hours.
        self.logger.info("====== MARKET OPEN: INITIATING TRADING SEQUENCE ======")
        
        self.streamer = LiveStreamer(self.api_key, self.secret_key, self.base_url, self.models_dir)
        stream_thread = threading.Thread(target=self.streamer.start_streaming, daemon=True)
        stream_thread.start()
        
        self.logger.info("Warming up Streamer Buffer (Waiting 60 seconds)...")
        time.sleep(60)
        
        while self.router.api.get_clock().is_open:
            matrix = self.streamer.get_latest_matrix()
            
            if matrix.empty or len(matrix) < 30:
                self.logger.info(f"[WAITING] Building live matrix buffer... ({len(matrix)}/30 ticks)")
                time.sleep(60)
                continue
                
            self.logger.info(f"Evaluating Matrix Shape: {matrix.shape} @ {pd.Timestamp.now().strftime('%H:%M:%S')}")
            
            signals, _ = generate_signals(matrix, self.models_dir)
            
            if not signals:
                self.logger.info("  -> No actionable signals generated this minute.")
            
            for spread_name, signal_data in signals.items():
                target_pos = signal_data['target_position']
                z_score = signal_data['current_z']
                
                position_id = f"{spread_name}_{target_pos}"
                
                if position_id not in self.active_positions:
                    action_str = 'LONG' if target_pos == 1 else 'SHORT'
                    self.logger.info(f"  *** [TRADE SIGNAL] {spread_name} | Z-Score: {z_score:.2f} | Action: {action_str}")
                    
                    if self.router.execute_spread(spread_name, signal_data, matrix):
                        self.active_positions.add(position_id)
                        
                        opposite_id = f"{spread_name}_{target_pos * -1}"
                        self.active_positions.discard(opposite_id)

            time.sleep(60)
            
        self.logger.info("Market is Closing. Terminating WebSocket connection.")
        if self.streamer and self.streamer.stream:
            self.streamer.stream.stop()
        self.active_positions.clear()

    def run_24_7_daemon(self):
        # The Master Infinite Loop that manages sleep cycles.
        self.logger.info("====== BOOTING QUANT 24/7 DAEMON ======")
        
        while True:
            try:
                clock = self.router.api.get_clock()
                
                if clock.is_open:
                    self._sync_m1_payload()
                    self._execute_trading_day()
                else:
                    self._sync_m1_payload()
                    
                    clock = self.router.api.get_clock()
                    time_to_open = (clock.next_open - clock.timestamp).total_seconds()
                    hours = time_to_open / 3600
                    
                    self.logger.info("Market is currently CLOSED.")
                    self.logger.info(f"Hibernating safely for {hours:.2f} hours. Will wake up automatically at market open.")
                    
                    time.sleep(time_to_open)
                    self.logger.info("ALARM: Market Open detected. Waking up from hibernation...")
                    
            except Exception as e:
                self.logger.error(f"CRITICAL DAEMON ERROR: {e}")
                self.logger.info("Attempting to recover in 60 seconds...")
                time.sleep(60)

if __name__ == "__main__":
    orchestrator = ExecutionOrchestrator()
    try:
        orchestrator.run_24_7_daemon()
    except KeyboardInterrupt:
        orchestrator.logger.info("24/7 Daemon Terminated by User via KeyboardInterrupt.")