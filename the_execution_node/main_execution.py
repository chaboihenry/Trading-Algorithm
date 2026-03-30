import os
import time
import threading
import subprocess
import pandas as pd
from dotenv import load_dotenv

from the_execution_node.core.live_streamer import LiveStreamer
from the_execution_node.core.order_router import OrderRouter
from the_execution_node.strategies.stat_arb_engine import generate_signals

class ExecutionOrchestrator:
    # The 24/7 Master Daemon of the Execution Node.
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")
        
        self.models_dir = "the_models"
        self.router = OrderRouter(self.api_key, self.secret_key, self.base_url)
        
        self.active_positions = set()
        self.streamer = None

    def _sync_m1_payload(self):
        # Automatically pulls the latest JSON research from GitHub
        print("\n[SYSTEM] Synchronizing with M1 Research Node (Git Pull)...")
        try:
            # Run git pull autonomously
            result = subprocess.run(
                ["git", "pull", "origin", "main"], 
                capture_output=True, text=True, check=True
            )
            print(f"[SUCCESS] GitHub Sync Complete: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] Git Pull failed (Offline?). Error: {e.stderr.strip()}")

    def _execute_trading_day(self):
        # The intra-day execution loop. Runs ONLY during market hours.
        print("\n====== MARKET OPEN: INITIATING TRADING SEQUENCE ======")
        
        # 1. Initialize a fresh streamer for the new day
        self.streamer = LiveStreamer(self.api_key, self.secret_key, self.base_url, self.models_dir)
        stream_thread = threading.Thread(target=self.streamer.start_streaming, daemon=True)
        stream_thread.start()
        
        print("[SYSTEM] Warming up Streamer Buffer (Waiting 60 seconds)...")
        time.sleep(60)
        
        # 2. Continuous Evaluation Loop (Breaks when market closes)
        while self.router.api.get_clock().is_open:
            matrix = self.streamer.get_latest_matrix()
            
            if matrix.empty or len(matrix) < 30:
                print(f"[WAITING] Building live matrix buffer... ({len(matrix)}/30 ticks)")
                time.sleep(60)
                continue
                
            print(f"\n[EVALUATING] Live Matrix Shape: {matrix.shape} @ {pd.Timestamp.now().strftime('%H:%M:%S')}")
            
            # 3. Ask the Brain for signals
            signals, _ = generate_signals(matrix, self.models_dir)
            
            if not signals:
                print("  -> No actionable signals generated this minute.")
            
            # 4. Route orders based on signals
            for spread_name, signal_data in signals.items():
                target_pos = signal_data['target_position']
                z_score = signal_data['current_z']
                
                position_id = f"{spread_name}_{target_pos}"
                
                if position_id not in self.active_positions:
                    print(f"  *** [TRADE SIGNAL] {spread_name} | Z-Score: {z_score:.2f} | Action: {'LONG' if target_pos == 1 else 'SHORT'}")
                    
                    if self.router.execute_spread(spread_name, signal_data, matrix):
                        self.active_positions.add(position_id)
                        
                        # Clear opposite direction state
                        opposite_id = f"{spread_name}_{target_pos * -1}"
                        self.active_positions.discard(opposite_id)

            # Evaluate exactly once per minute
            time.sleep(60)
            
        # 5. Market Close Teardown
        print("\n[SYSTEM] Market is Closing. Terminating WebSocket connection.")
        if self.streamer and self.streamer.stream:
            self.streamer.stream.stop()
        self.active_positions.clear()

    def run_24_7_daemon(self):
        # The Master Infinite Loop that manages sleep cycles.
        print("\n====== BOOTING QUANT 24/7 DAEMON ======")
        
        while True:
            clock = self.router.api.get_clock()
            
            if clock.is_open:
                # If we start the script in the middle of the trading day
                self._sync_m1_payload()
                self._execute_trading_day()
            else:
                # If market is closed, sync data and calculate exact sleep time
                self._sync_m1_payload()
                
                # Fetch fresh clock to be absolutely precise
                clock = self.router.api.get_clock()
                time_to_open = (clock.next_open - clock.timestamp).total_seconds()
                
                hours = time_to_open / 3600
                print(f"\n[HIBERNATION] Market is currently CLOSED.")
                print(f"[HIBERNATION] Sleeping safely for {hours:.2f} hours. Will wake up automatically at market open.")
                
                # The script will completely freeze here, consuming ~0% CPU, until the exact second the market opens
                time.sleep(time_to_open)
                
                print("\n[SYSTEM] ALARM: Market Open detected. Waking up from hibernation...")

if __name__ == "__main__":
    orchestrator = ExecutionOrchestrator()
    try:
        orchestrator.run_24_7_daemon()
    except KeyboardInterrupt:
        print("\n[SYSTEM] 24/7 Daemon Terminated by User.")