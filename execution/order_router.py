# Alpaca (and most brokers) do not allow you to short-sell using dollar amounts (notional orders). 
# MUST specify the exact fractional quantity of shares you want to short!

import os 
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
import math

class Alpacarouter:
    def __init__(self, paper: bool = True):
        # load keys from .env file
        load_dotenv()
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise ValueError("Critical: Alpaca keys not found in environment.")
        
        # initialize the trading and data clients
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        print(f"Executor: Alpaca Trading Client Initalized (Paper mode: {paper})")

    def get_live_price(self, symbol:str): 
        # Fetches the most recent bid/ask midpoint to calculate share quantities
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = self.data_client.get_stock_latest_quote(request)[symbol]
        # use midpoint of spread for sizing
        return (quote.ask_price + quote.bid_price) / 2.0
    
    def calculate_leg_quantities(self, spread_allocation: float, weights: dict, target_position: int):
        # translates spread's dollar allocation into exact fractional share quantities
        # target_position: 1 (long the spread), -1 (short the spread)
        total_abs_weight = sum(abs(w) for w in weights.values())
        
        orders = {}
        for ticker, weight in weights.items():
            leg_dollar_allocation = (abs(weight) / total_abs_weight) * spread_allocation
            live_price = self.get_live_price(ticker)
            
            raw_qty = leg_dollar_allocation / live_price
            
            directional_weight = weight * target_position
            side = OrderSide.BUY if directional_weight > 0 else OrderSide.SELL
            
            # enforce the physical limits of short selling
            if side == OrderSide.SELL:
                # cannot borrow a fraction of a share. Must be a whole integer.
                qty = math.floor(raw_qty)
                
                # THE KILL SWITCH: If you can't afford 1 whole share, kill the spread
                if qty == 0:
                    print(f"\n[WARNING] Insufficient allocated capital to short 1 whole share of {ticker} (Price: ${live_price:.2f}).")
                    print("[WARNING] Aborting entire spread to prevent an unhedged directional trap.")
                    return {} # Returning an empty dict ensures NO orders are routed
            else:
                # long orders can remain fractional
                qty = round(raw_qty, 4)
            
            orders[ticker] = {
                'qty': qty,
                'side': side,
                'notional_value': leg_dollar_allocation # Note: theoretical notional, not actual executed
            }
            
        return orders
    
    def execute_spread(self, spread_name: str, orders: dict):
        # fires the calc orders into the market
        print(f"\n---Routing Orders for {spread_name} ---")
        for ticker, details in orders.items():
            if details['qty'] <= 0.0001:
                print(f"Skipping {ticker} - Quantity too small: {details['qty']}")
                continue

            print(f"Submitting: {details['side'].name} {details['qty']} shares of {ticker} (~${details['notional_value']:.2f})")

            req = MarketOrderRequest(
                symbol=ticker,
                qty=details['qty'], 
                side=details['side'],
                time_in_force=TimeInForce.DAY
            )

            # send the order
            try:
                self.trading_client.submit_order(order_data=req)
            except Exception as e:
                print(f"[ERROR] Failed to execute {ticker}: {e}")

    def liquidate_portfolio(self):
        # Emergency kill-swtich from the Shield
        print("\nEmergency Override: Liquidating all open positions and canceling pending orders...")
        self.trading_client.cancel_orders()
        self.trading_client.close_all_positions(cancel_orders=True)
        print("Success: Portfolio is Flat. Cash secured.")


if __name__ == "__main__":
    print("=== Testing Alpaca Execution Router ===")

    # instantiate in paper trading mode to protect real capital
    router = Alpacarouter(paper=True)

    # mocking data from 'the_filer' and 'the_allocator'
    mock_spread_allocation = 4.65        # The HRP budget for this spread
    mock_johansen_weights = {'V': 1.0, 'MA': -0.5412}
    mock_target_position = 1             # +1 means Long the spread (Buy V, Short MA)

    print(f"\nIncoming Spread Allocation: ${mock_spread_allocation:.2f}")

    # 1. Translate dollars to shares 
    leg_orders = router.calculate_leg_quantities(
        spread_allocation=mock_spread_allocation,
        weights=mock_johansen_weights,
        target_position=mock_target_position
    )

    # 2. Execute the trades
    router.execute_spread('V_MA_Spread', leg_orders)
    
    for ticker, details in leg_orders.items():
        print(f"Calculated Leg -> {ticker}: {details['side'].name} {details['qty']} shares")

