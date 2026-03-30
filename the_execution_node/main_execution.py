import os
import time
import threading
import pandas as pd
from dotenv import load_dotenv

from core.live_streamer import LiveStreamer
from core.order_router import OrderRouter
from strategies.stat_arb_engine import generate_signals

class ExecutionOrchestrator:
    # The Master Loop of the Execution Node.
    # Weaves the Streamer (Eyes), Router (Hands), and Stat-Arb Engine (Brain) together.
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")
        
        self.models_dir = "the_models"
        
        # 1. Initialize Core Components
        self.streamer = LiveStreamer(self.api_key, self.secret_key, self.base_url, self.models_dir)
        self.router = OrderRouter(self.api_key, self.secret_key, self.base_url)
        
        # 2. Track active positions to prevent spamming duplicate orders
        self.active_positions = set()

    def run(self):
        # The main autonomous execution loop.
        print("\n====== STARTING QUANT EXECUTION NODE ======")
        
        # 1. Start Streamer in the background
        stream_thread = threading.Thread(target=self.streamer.start_streaming, daemon=True)
        stream_thread.start()
        
        print("[SYSTEM] Warming up Streamer Buffer (Waiting 60 seconds)...")
        time.sleep(60)
        
        # 2. Infinite Evaluation Loop
        while True:
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
                
                # Create a unique state ID (e.g., "AAPL_MSFT_Spread_1" for LONG)
                position_id = f"{spread_name}_{target_pos}"
                
                if position_id not in self.active_positions:
                    print(f"  *** [TRADE SIGNAL] {spread_name} | Z-Score: {z_score:.2f} | Action: {'LONG' if target_pos == 1 else 'SHORT'}")
                    
                    # 5. Execute the trade
                    if self.router.execute_spread(spread_name, signal_data, matrix):
                        self.active_positions.add(position_id)
                        
                        # If we went long, clear any record of being short, and vice versa
                        opposite_id = f"{spread_name}_{target_pos * -1}"
                        self.active_positions.discard(opposite_id)

            # Sleep until the next minute bar arrives
            time.sleep(60)

if __name__ == "__main__":
    orchestrator = ExecutionOrchestrator()
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Execution Node Terminated by User.")