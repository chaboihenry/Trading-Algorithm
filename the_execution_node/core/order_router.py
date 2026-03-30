import os
import math
import pandas as pd
from alpaca_trade_api.rest import REST

class OrderRouter:
    # Executes trades based on HRP allocations and Strategy Engine signals.
    # Acts as 'The Shield' by verifying buying power before routing.
    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.api = REST(api_key, secret_key, base_url)

    def get_account_metrics(self):
        # Fetches live account equity and buying power.
        account = self.api.get_account()
        return float(account.equity), float(account.buying_power)

    def execute_spread(self, spread_name: str, signal_data: dict, live_matrix: pd.DataFrame):
        # Calculates position sizing and routes orders for a cointegrated basket.
        equity, buying_power = self.get_account_metrics()
        
        target_pos = signal_data.get('target_position', 0)
        allocation = signal_data.get('hrp_allocation', 0.0)
        weights = signal_data.get('johansen_weights', {})
        
        # 1. Calculate capital limit for this specific spread
        spread_capital = equity * allocation
        
        # SHIELD: Do not execute if out of buying power
        if spread_capital > buying_power:
            print(f"[SHIELD] Blocked {spread_name}: Required ${spread_capital:.2f}, Available ${buying_power:.2f}")
            return False

        print(f"\n[ROUTER] Routing {spread_name} | Allocation: {allocation*100:.2f}% (${spread_capital:.2f})")
        
        # 2. Divide capital equally among the legs (Dollar Neutrality)
        num_legs = len(weights)
        if num_legs == 0:
            print(f"[WARNING] No weights found for {spread_name}.")
            return False
            
        leg_capital = spread_capital / num_legs
        
        # 3. Execute each leg based on statistical arbitrage weights
        for ticker, weight in weights.items():
            if ticker not in live_matrix.columns or live_matrix.empty:
                print(f"  -> [WARNING] No live price for {ticker}. Skipping leg.")
                continue
                
            # Get the most recent price from the matrix
            price = live_matrix[ticker].iloc[-1]
            if pd.isna(price) or price <= 0:
                print(f"  -> [WARNING] Invalid live price for {ticker}. Skipping leg.")
                continue
                
            # Direction Logic: Target Pos (1 or -1) * Johansen Weight (Pos or Neg)
            # Example: Going long (1) a negative weight (-0.5) results in a SHORT (-0.5).
            trade_direction = target_pos * weight
            side = 'buy' if trade_direction > 0 else 'sell'
            
            # Size position (Floor to integer to avoid fractional share API rejections)
            qty = math.floor(leg_capital / price)
            
            if qty <= 0:
                print(f"  -> [WARNING] Insufficient capital to {side.upper()} 1 share of {ticker}.")
                continue
                
            try:
                self.api.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side=side,
                    type='market',
                    time_in_force='day'
                )
                print(f"  -> [EXECUTED] {side.upper()} {qty}x {ticker} @ ~${price:.2f}")
            except Exception as e:
                print(f"  -> [ERROR] Alpaca API rejected {ticker}: {e}")
                
        return True


if __name__ == "__main__":
    from dotenv import load_dotenv

    print("\n====== EXECUTION NODE DIAGNOSTIC: ORDER ROUTER ======")

    # 1. Load environment variables for the diagnostic run
    load_dotenv()
    API_KEY = os.getenv("ALPACA_API_KEY")
    SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    BASE_URL = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")

    if not API_KEY or not SECRET_KEY:
        print("[CRITICAL] Alpaca API keys not found in .env file.")
        exit(1)

    print("[SYSTEM] Environment variables loaded. Connecting to Alpaca...")

    # 2. Initialize Router
    router = OrderRouter(API_KEY, SECRET_KEY, BASE_URL)
    
    try:
        equity, bp = router.get_account_metrics()
        print(f"[SUCCESS] Connected to Alpaca.")
        print(f"  -> Account Equity: ${equity:.2f}")
        print(f"  -> Buying Power:   ${bp:.2f}")
        
        # 3. Diagnostic Test 1: LONG Mechanics
        print("\n[SYSTEM] Testing LONG mechanics (Buy 1 AAPL @ $1.00 Limit)...")
        long_order = router.api.submit_order(
            symbol='AAPL',
            qty=1,
            side='buy',
            type='limit',
            time_in_force='day',
            limit_price=1.00
        )
        print(f"[SUCCESS] LONG Order routed successfully! ID: {long_order.id}")

        # 4. Diagnostic Test 2: SHORT Mechanics (Changed to MSFT to avoid position conflict)
        print("\n[SYSTEM] Testing SHORT mechanics (Sell 1 MSFT @ $5000.00 Limit)...")
        short_order = router.api.submit_order(
            symbol='MSFT',
            qty=1,
            side='sell',
            type='limit',
            time_in_force='day',
            limit_price=5000.00
        )
        print(f"[SUCCESS] SHORT Order routed successfully! ID: {short_order.id}")

        print("\n====== DIAGNOSTIC RESULTS ======")
        print("[ACTION] Check your Alpaca Dashboard (Orders Tab).")
        print("[ACTION] You should see one pending BUY and one pending SELL for AAPL.")
        print("[ACTION] Both are deeply out-of-the-money limit orders and are 100% safe.")
        print("[SYSTEM] Diagnostic complete.")
        
    except Exception as e:
        print(f"\n[CRITICAL FAILURE] Connection or Routing failed: {e}")