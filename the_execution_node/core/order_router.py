import os
import math
import json
import pandas as pd
from alpaca_trade_api.rest import REST

from the_utilities.strategy_config import LEVERAGE, SLIPPAGE_BPS_LIVE, HEDGE_RATIO_MIN


class OrderRouter:
    # Executes trades based on HRP allocations, Half-Kelly sizing,
    # and Johansen hedge ratios. Verifies buying power before routing.

    SLIPPAGE_BPS = SLIPPAGE_BPS_LIVE

    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.api = REST(api_key, secret_key, base_url)

    def get_account_metrics(self):
        # Fetches live account equity and buying power
        account = self.api.get_account()
        return float(account.equity), float(account.buying_power)

    def get_open_positions(self):
        # Returns a set of tickers currently held in the account
        try:
            positions = self.api.list_positions()
            return {p.symbol for p in positions}
        except Exception as e:
            print(f"[WARNING] Could not fetch positions: {e}")
            return set()

    def execute_spread(self, spread_name: str, signal_data: dict, live_matrix: pd.DataFrame, open_spreads: set):
        # Routes a cointegrated basket trade with proper hedge ratio sizing
        # Returns a dict: {"success": bool, "entry_prices": {...}, "leg_shares": {...}}
        # entry_prices and leg_shares are needed for P&L logging on close

        # 1. Block duplicate entries — never stack positions on the same spread
        if spread_name in open_spreads:
            print(f"[ROUTER] Blocked {spread_name}: already holding this spread.")
            return {"success": False}

        equity, buying_power = self.get_account_metrics()

        target_pos = signal_data.get('target_position', 0)
        allocation = signal_data.get('hrp_allocation', 0.0)
        weights = signal_data.get('johansen_weights', {})
        bet_size = signal_data.get('bet_size', 0.5)

        if target_pos == 0 or not weights:
            print(f"[WARNING] No actionable signal for {spread_name}.")
            return {"success": False}

        # 2. Calculate capital for this spread using HRP allocation and Half-Kelly
        spread_capital = equity * allocation * bet_size * LEVERAGE

        # Shield: block if insufficient buying power
        if spread_capital > buying_power:
            print(f"[SHIELD] Blocked {spread_name}: "
                  f"Required ${spread_capital:.2f}, Available ${buying_power:.2f}")
            return {"success": False}

        print(f"\n[ROUTER] Routing {spread_name} | "
              f"Capital: ${spread_capital:.2f} | Bet Size: {bet_size:.3f}")

        # 3. Size each leg proportional to Johansen weight magnitudes
        # Normalize weights so their absolute values sum to 1
        abs_weight_sum = sum(abs(w) for w in weights.values())
        if abs_weight_sum == 0:
            print(f"[WARNING] All weights are zero for {spread_name}.")
            return {"success": False}

        held_tickers = self.get_open_positions()

        # Pre-check: verify at least 1 share of every leg is affordable
        # Prevents log spam from spreads with high share prices and low allocation
        for ticker, weight in weights.items():
            if ticker not in live_matrix.columns or live_matrix.empty:
                print(f"[ROUTER] Blocked {spread_name}: no live price for {ticker}.")
                return {"success": False}

            price = live_matrix[ticker].iloc[-1]
            if pd.isna(price) or price <= 0:
                continue

            weight_fraction = abs(weight) / abs_weight_sum
            leg_capital = spread_capital * weight_fraction

            if leg_capital < price:
                print(f"[ROUTER] Blocked {spread_name}: "
                      f"allocation ${spread_capital:.0f} can't buy 1 share of {ticker} (${price:.0f}).")
                return {"success": False}

        # Pre-check: reject degenerate spreads where one leg barely hedges the other
        # A "hedge" with <HEDGE_RATIO_MIN of the largest leg is essentially a naked directional bet
        leg_notionals = []
        for ticker, weight in weights.items():
            if ticker not in live_matrix.columns:
                continue
            price = live_matrix[ticker].iloc[-1]
            if pd.isna(price) or price <= 0:
                continue
            wt_frac = abs(weight) / abs_weight_sum
            leg_notionals.append(spread_capital * wt_frac)

        if leg_notionals:
            min_leg = min(leg_notionals)
            max_leg = max(leg_notionals)
            if max_leg > 0 and (min_leg / max_leg) < HEDGE_RATIO_MIN:
                hedge_ratio_pct = 100 * min_leg / max_leg
                print(f"[ROUTER] Blocked {spread_name}: "
                      f"degenerate hedge (smallest leg is {hedge_ratio_pct:.1f}% of largest).")
                return {"success": False}

        # Pre-check: verify every short leg is actually shortable on Alpaca
        # Prevents partial-fill unhedged exposure from non-shortable assets like SO
        for ticker, weight in weights.items():
            trade_direction = target_pos * weight
            if trade_direction < 0:  # This leg would be a short sale
                try:
                    asset = self.api.get_asset(ticker)
                    if not asset.shortable:
                        print(f"[ROUTER] Blocked {spread_name}: {ticker} is not shortable.")
                        return {"success": False}
                except Exception as e:
                    print(f"[ROUTER] Blocked {spread_name}: could not verify {ticker} shortability ({e}).")
                    return {"success": False}

        # 4. Execute each leg and track entry prices/shares for P&L logging
        # Only successfully filled legs are recorded — partial fills won't pollute analytics
        entry_prices = {}
        leg_shares = {}
        executed_legs = 0
        total_legs = len(weights)

        for ticker, weight in weights.items():
            # Skip if ticker is already held in another position to avoid conflicts
            if ticker in held_tickers:
                print(f"[WARNING] Already holding {ticker}. Skipping to avoid conflict.")
                continue

            if ticker not in live_matrix.columns or live_matrix.empty:
                print(f"[WARNING] No live price for {ticker}. Skipping leg.")
                continue

            price = live_matrix[ticker].iloc[-1]
            if pd.isna(price) or price <= 0:
                print(f"[WARNING] Invalid live price for {ticker}. Skipping leg.")
                continue

            # Capital for this leg proportional to its weight magnitude
            weight_fraction = abs(weight) / abs_weight_sum
            leg_capital = spread_capital * weight_fraction

            # Direction: target_pos (1/-1) * johansen weight sign
            trade_direction = target_pos * weight
            side = 'buy' if trade_direction > 0 else 'sell'

            # Floor to integer shares
            qty = math.floor(leg_capital / price)

            if qty <= 0:
                print(f"[WARNING] Insufficient capital for 1 share of {ticker} "
                      f"(need ${price:.2f}, have ${leg_capital:.2f}).")
                continue

            try:
                order = self.api.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side=side,
                    type='market',
                    time_in_force='day'
                )
                notional = qty * price
                print(f"[EXECUTED] {side.upper()} {qty}x {ticker} "
                      f"@ ~${price:.2f} (${notional:.2f}) | Order: {order.id}")

                # Record entry data for P&L logging at close
                # Signed shares: positive for long, negative for short
                signed_qty = qty if side == 'buy' else -qty
                entry_prices[ticker] = float(price)
                leg_shares[ticker] = signed_qty
                executed_legs += 1
            except Exception as e:
                print(f"[ERROR] Alpaca API rejected {ticker}: {e}")

        # 5. Validate execution outcome
        if executed_legs == 0:
            print(f"[CRITICAL] No legs executed for {spread_name}. Trade aborted.")
            return {"success": False}
        elif executed_legs < total_legs:
            print(f"[WARNING] Partial execution for {spread_name}: "
                  f"{executed_legs}/{total_legs} legs filled. Hedge is incomplete.")

        return {
            "success": True,
            "entry_prices": entry_prices,
            "leg_shares": leg_shares,
        }

    def close_spread(self, spread_name: str, weights: dict, reason: str, position_data: dict = None):
        # Closes all legs and logs the trade to CSV for post-trade analysis
        # position_data carries entry prices, shares, entry_z, etc., from main_execution state
        from datetime import datetime
        from the_utilities.trade_logger import log_trade

        print(f"\n[ROUTER] Closing {spread_name} | Reason: {reason}")

        held_tickers = self.get_open_positions()
        closed = 0
        exit_prices = {}

        # Capture exit prices BEFORE closing — close_position is async, so quote it now
        for ticker in weights:
            if ticker not in held_tickers:
                continue
            try:
                quote = self.api.get_latest_trade(ticker)
                exit_prices[ticker] = float(quote.price)
            except Exception as e:
                print(f"[WARNING] Could not fetch exit price for {ticker}: {e}")
                exit_prices[ticker] = 0.0

        # Submit close orders
        for ticker in weights:
            if ticker not in held_tickers:
                continue
            try:
                self.api.close_position(ticker)
                print(f"[CLOSED] {ticker}")
                closed += 1
            except Exception as e:
                print(f"[ERROR] Failed to close {ticker}: {e}")

        if closed == 0:
            print(f"[INFO] No positions found to close for {spread_name}.")
            return False

        # Log trade only if full metadata from caller is available
        if position_data and "leg_shares" in position_data and "entry_prices" in position_data:
            entry_prices = position_data["entry_prices"]
            leg_shares = position_data["leg_shares"]

            gross_pnl = sum(
                shares * (exit_prices.get(ticker, entry_prices.get(ticker, 0))
                          - entry_prices.get(ticker, 0))
                for ticker, shares in leg_shares.items()
            )

            slippage_cost = sum(
                abs(shares) * (entry_prices.get(ticker, 0) + exit_prices.get(ticker, 0))
                * (self.SLIPPAGE_BPS / 10000.0)
                for ticker, shares in leg_shares.items()
            )

            net_pnl = gross_pnl - slippage_cost
            capital_allocated = sum(
                abs(shares) * entry_prices.get(ticker, 0)
                for ticker, shares in leg_shares.items()
            )
            pnl_pct = (net_pnl / capital_allocated * 100) if capital_allocated > 0 else 0.0

            log_trade({
                "exit_timestamp": datetime.now().isoformat(),
                "spread_name": spread_name,
                "direction": "LONG" if position_data.get("target_position") == 1 else "SHORT",
                "entry_timestamp": position_data.get("entry_timestamp", ""),
                "bars_held": position_data.get("bars_held", 0),
                "exit_reason": reason,
                "entry_z": round(position_data.get("entry_z", 0.0), 4),
                "exit_z": round(position_data.get("current_z", 0.0), 4),
                "ai_confidence": round(position_data.get("ai_confidence", 0.0), 4),
                "bet_size": round(position_data.get("bet_size", 0.0), 4),
                "capital_allocated": round(capital_allocated, 2),
                "tickers": "|".join(leg_shares.keys()),
                "entry_prices": "|".join(f"{entry_prices.get(t, 0):.4f}" for t in leg_shares),
                "exit_prices": "|".join(f"{exit_prices.get(t, 0):.4f}" for t in leg_shares),
                "shares": "|".join(str(s) for s in leg_shares.values()),
                "gross_pnl": round(gross_pnl, 2),
                "slippage_cost": round(slippage_cost, 2),
                "net_pnl": round(net_pnl, 2),
                "pnl_pct": round(pnl_pct, 4),
            })

        return True

    def cancel_all_open_orders(self):
        # Emergency cleanup — cancels every pending order in the account
        try:
            self.api.cancel_all_orders()
            print("[ROUTER] All open orders cancelled.")
        except Exception as e:
            print(f"[ERROR] Failed to cancel orders: {e}")

    def emergency_liquidate(self):
        # Nuclear option — closes every position and cancels every order
        print("[EMERGENCY] Liquidating all positions and cancelling all orders...")
        self.cancel_all_open_orders()
        try:
            self.api.close_all_positions()
            print("[EMERGENCY] All positions closed.")
        except Exception as e:
            print(f"[ERROR] Emergency liquidation failed: {e}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    print("\n====== EXECUTION NODE DIAGNOSTIC: ORDER ROUTER ======")

    load_dotenv()
    API_KEY = os.getenv("ALPACA_API_KEY")
    SECRET_KEY = os.getenv("ALPACA_API_SECRET")
    BASE_URL = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")

    if not API_KEY or not SECRET_KEY:
        print("[CRITICAL] Alpaca API keys not found in .env file.")
        exit(1)

    print("[SYSTEM] Environment variables loaded. Connecting to Alpaca...")

    router = OrderRouter(API_KEY, SECRET_KEY, BASE_URL)

    try:
        equity, bp = router.get_account_metrics()
        print(f"[SUCCESS] Connected to Alpaca.")
        print(f"Account Equity: ${equity:.2f}")
        print(f"Buying Power:   ${bp:.2f}")

        positions = router.get_open_positions()
        print(f"Open Positions: {positions if positions else 'None'}")

        # Diagnostic: submit a safe out-of-the-money limit order
        print("\n[SYSTEM] Testing LONG mechanics (Buy 1 AAPL @ $1.00 Limit)...")
        long_order = router.api.submit_order(
            symbol='AAPL',
            qty=1,
            side='buy',
            type='limit',
            time_in_force='day',
            limit_price=1.00
        )
        print(f"[SUCCESS] LONG order routed. ID: {long_order.id}")

        print("\n[SYSTEM] Testing SHORT mechanics (Sell 1 MSFT @ $5000.00 Limit)...")
        short_order = router.api.submit_order(
            symbol='MSFT',
            qty=1,
            side='sell',
            type='limit',
            time_in_force='day',
            limit_price=5000.00
        )
        print(f"[SUCCESS] SHORT order routed. ID: {short_order.id}")

        # Clean up diagnostic orders
        print("\n[SYSTEM] Cancelling diagnostic orders...")
        router.cancel_all_open_orders()

        print("\n====== DIAGNOSTIC COMPLETE ======")

    except Exception as e:
        print(f"\n[CRITICAL FAILURE] Connection or Routing failed: {e}")