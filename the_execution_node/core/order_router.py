import os
import math
import json
import logging
import pandas as pd
from alpaca_trade_api.rest import REST

from the_utilities.strategy_config import LEVERAGE, SLIPPAGE_BPS_LIVE, HEDGE_RATIO_MIN

logger = logging.getLogger(__name__)


class OrderRouter:
    # Executes trades based on HRP allocations, Half-Kelly sizing,
    # and Johansen hedge ratios. Verifies buying power before routing.

    SLIPPAGE_BPS = SLIPPAGE_BPS_LIVE

    def __init__(self, api_key: str, secret_key: str, base_url: str, logger=None):
        self.api = REST(api_key, secret_key, base_url)
        self.logger = logger if logger is not None else logging.getLogger(__name__)

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
            self.logger.warning(f"[WARNING] Could not fetch positions: {e}")
            return set()

    def execute_spread(self, spread_name: str, signal_data: dict, live_matrix: pd.DataFrame, open_spreads: set):
        # Routes a cointegrated basket trade with proper hedge ratio sizing
        # Returns a dict: {"success": bool, "entry_prices": {...}, "leg_shares": {...}}
        # entry_prices and leg_shares are needed for P&L logging on close

        # 1. Block duplicate entries — never stack positions on the same spread
        if spread_name in open_spreads:
            self.logger.warning(f"[ROUTER] Blocked {spread_name}: already holding this spread.")
            return {"success": False, "reason": "duplicate_spread"}

        equity, buying_power = self.get_account_metrics()

        target_pos = signal_data.get('target_position', 0)
        allocation = signal_data.get('hrp_allocation', 0.0)
        weights = signal_data.get('johansen_weights', {})
        bet_size = signal_data.get('bet_size', 0.5)

        if target_pos == 0 or not weights:
            self.logger.warning(f"[WARNING] No actionable signal for {spread_name}.")
            return {"success": False, "reason": "no_signal"}

        # 2. Calculate capital for this spread using HRP allocation and Half-Kelly
        spread_capital = equity * allocation * bet_size * LEVERAGE

        # Shield: block if insufficient buying power
        if spread_capital > buying_power:
            self.logger.warning(
                f"[SHIELD] Blocked {spread_name}: "
                f"Required ${spread_capital:.2f}, Available ${buying_power:.2f}"
            )
            return {"success": False, "reason": "insufficient_capital"}

        self.logger.info(
            f"[ROUTER] Routing {spread_name} | "
            f"Capital: ${spread_capital:.2f} | Bet Size: {bet_size:.3f}"
        )

        # 3. Size each leg proportional to Johansen weight magnitudes
        # Normalize weights so their absolute values sum to 1
        abs_weight_sum = sum(abs(w) for w in weights.values())
        if abs_weight_sum == 0:
            self.logger.warning(f"[WARNING] All weights are zero for {spread_name}.")
            return {"success": False, "reason": "zero_weights"}

        # ── Phase 1: pre-flight validation (no orders submitted) ──────────────
        # Every leg must pass every check before any order is submitted.
        # A single failure rejects the entire basket to preserve market neutrality.

        held_tickers = self.get_open_positions()
        leg_notionals = []  # accumulated for degenerate-hedge check below

        for ticker, weight in weights.items():
            # Live price must be present and valid
            if ticker not in live_matrix.columns or live_matrix.empty:
                self.logger.warning(f"[ROUTER] Blocked {spread_name}: no live price for {ticker}.")
                return {"success": False, "reason": f"no_price:{ticker}"}

            price = live_matrix[ticker].iloc[-1]
            if pd.isna(price) or price <= 0:
                self.logger.warning(f"[WARNING] Invalid live price for {ticker}. Skipping leg.")
                return {"success": False, "reason": f"invalid_price:{ticker}"}

            # Ticker must not already be held in any open Alpaca position
            if ticker in held_tickers:
                self.logger.warning(f"[WARNING] Already holding {ticker}. Skipping to avoid conflict.")
                return {"success": False, "reason": f"already_held:{ticker}"}

            weight_fraction = abs(weight) / abs_weight_sum
            leg_capital = spread_capital * weight_fraction

            # Must be able to afford at least 1 share
            if leg_capital < price:
                self.logger.warning(
                    f"[ROUTER] Blocked {spread_name}: "
                    f"allocation ${spread_capital:.0f} can't buy 1 share of {ticker} (${price:.0f})."
                )
                return {"success": False, "reason": f"cant_afford:{ticker}"}

            qty = math.floor(leg_capital / price)
            if qty <= 0:
                self.logger.warning(
                    f"[WARNING] Insufficient capital for 1 share of {ticker} "
                    f"(need ${price:.2f}, have ${leg_capital:.2f})."
                )
                return {"success": False, "reason": f"qty_zero:{ticker}"}

            # Short legs must be shortable on Alpaca
            # Prevents partial-fill unhedged exposure from non-shortable assets like SO
            if target_pos * weight < 0:
                try:
                    asset = self.api.get_asset(ticker)
                    if not asset.shortable:
                        self.logger.warning(f"[ROUTER] Blocked {spread_name}: {ticker} is not shortable.")
                        return {"success": False, "reason": f"not_shortable:{ticker}"}
                except Exception as e:
                    self.logger.warning(
                        f"[ROUTER] Blocked {spread_name}: could not verify {ticker} shortability ({e})."
                    )
                    return {"success": False, "reason": f"shortability_check_failed:{ticker}"}

            leg_notionals.append(leg_capital)

        # Reject degenerate spreads where one leg barely hedges the other
        # A "hedge" with <HEDGE_RATIO_MIN of the largest leg is essentially a naked directional bet
        if leg_notionals:
            min_leg = min(leg_notionals)
            max_leg = max(leg_notionals)
            if max_leg > 0 and (min_leg / max_leg) < HEDGE_RATIO_MIN:
                hedge_ratio_pct = 100 * min_leg / max_leg
                self.logger.warning(
                    f"[ROUTER] Blocked {spread_name}: "
                    f"degenerate hedge (smallest leg is {hedge_ratio_pct:.1f}% of largest)."
                )
                return {"success": False, "reason": "degenerate_hedge"}

        # ── Phase 2: order submission ──────────────────────────────────────────
        # All legs passed pre-flight. Submit every leg; no eligibility skips here.

        entry_prices = {}
        leg_shares = {}
        executed_legs = 0
        total_legs = len(weights)

        for ticker, weight in weights.items():
            price = live_matrix[ticker].iloc[-1]
            weight_fraction = abs(weight) / abs_weight_sum
            leg_capital = spread_capital * weight_fraction
            trade_direction = target_pos * weight
            side = 'buy' if trade_direction > 0 else 'sell'
            qty = math.floor(leg_capital / price)

            try:
                order = self.api.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side=side,
                    type='market',
                    time_in_force='day'
                )
                notional = qty * price
                self.logger.info(
                    f"[EXECUTED] {side.upper()} {qty}x {ticker} "
                    f"@ ~${price:.2f} (${notional:.2f}) | Order: {order.id}"
                )

                # Record entry data for P&L logging at close
                # Signed shares: positive for long, negative for short
                signed_qty = qty if side == 'buy' else -qty
                entry_prices[ticker] = float(price)
                leg_shares[ticker] = signed_qty
                executed_legs += 1
            except Exception as e:
                self.logger.error(f"[ERROR] Alpaca API rejected {ticker}: {e}")

        # Guard against broker-side rejections (e.g. Alpaca API error mid-basket)
        if executed_legs == 0:
            self.logger.error(f"[CRITICAL] No legs executed for {spread_name}. Trade aborted.")
            return {"success": False, "reason": "no_legs_executed"}
        elif executed_legs < total_legs:
            self.logger.warning(
                f"[WARNING] Partial execution for {spread_name}: "
                f"{executed_legs}/{total_legs} legs filled. Hedge is incomplete."
            )

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

        self.logger.info(f"[ROUTER] Closing {spread_name} | Reason: {reason}")

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
                self.logger.warning(f"[WARNING] Could not fetch exit price for {ticker}: {e}")
                exit_prices[ticker] = 0.0

        # Submit close orders
        for ticker in weights:
            if ticker not in held_tickers:
                continue
            try:
                self.api.close_position(ticker)
                self.logger.info(f"[CLOSED] {ticker}")
                closed += 1
            except Exception as e:
                self.logger.error(f"[ERROR] Failed to close {ticker}: {e}")

        if closed == 0:
            self.logger.info(f"[INFO] No positions found to close for {spread_name}.")
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
            self.logger.info("[ROUTER] All open orders cancelled.")
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to cancel orders: {e}")

    def emergency_liquidate(self):
        # Nuclear option — closes every position and cancels every order
        self.logger.warning("[EMERGENCY] Liquidating all positions and cancelling all orders...")
        self.cancel_all_open_orders()
        try:
            self.api.close_all_positions()
            self.logger.warning("[EMERGENCY] All positions closed.")
        except Exception as e:
            self.logger.error(f"[ERROR] Emergency liquidation failed: {e}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("====== EXECUTION NODE DIAGNOSTIC: ORDER ROUTER ======")

    load_dotenv()
    API_KEY = os.getenv("ALPACA_API_KEY")
    SECRET_KEY = os.getenv("ALPACA_API_SECRET")
    BASE_URL = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")

    if not API_KEY or not SECRET_KEY:
        logger.error("[CRITICAL] Alpaca API keys not found in .env file.")
        exit(1)

    logger.info("[SYSTEM] Environment variables loaded. Connecting to Alpaca...")

    router = OrderRouter(API_KEY, SECRET_KEY, BASE_URL)

    try:
        equity, bp = router.get_account_metrics()
        logger.info(f"[SUCCESS] Connected to Alpaca.")
        logger.info(f"Account Equity: ${equity:.2f}")
        logger.info(f"Buying Power:   ${bp:.2f}")

        positions = router.get_open_positions()
        logger.info(f"Open Positions: {positions if positions else 'None'}")

        # Diagnostic: submit a safe out-of-the-money limit order
        logger.info("[SYSTEM] Testing LONG mechanics (Buy 1 AAPL @ $1.00 Limit)...")
        long_order = router.api.submit_order(
            symbol='AAPL',
            qty=1,
            side='buy',
            type='limit',
            time_in_force='day',
            limit_price=1.00
        )
        logger.info(f"[SUCCESS] LONG order routed. ID: {long_order.id}")

        logger.info("[SYSTEM] Testing SHORT mechanics (Sell 1 MSFT @ $5000.00 Limit)...")
        short_order = router.api.submit_order(
            symbol='MSFT',
            qty=1,
            side='sell',
            type='limit',
            time_in_force='day',
            limit_price=5000.00
        )
        logger.info(f"[SUCCESS] SHORT order routed. ID: {short_order.id}")

        # Clean up diagnostic orders
        logger.info("[SYSTEM] Cancelling diagnostic orders...")
        router.cancel_all_open_orders()

        logger.info("====== DIAGNOSTIC COMPLETE ======")

    except Exception as e:
        logger.error(f"[CRITICAL FAILURE] Connection or Routing failed: {e}")