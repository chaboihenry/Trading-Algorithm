import time
import pandas as pd
import xgboost as xgb
from datetime import datetime

# import the architectural engines
from the_oracle.live_feed import MarketOracle
from the_shield.regime_detector import check_macro_safety
from the_brain.cointegration_arbitrage.stat_arb_engine import generate_signals
from the_filter.xgboost_meta_labeler import calculate_kelly_size
from the_allocator.hrp_engine import allocate_capital
from execution.order_router import Alpacarouter
import json

def run_trading_cycle():
    print(f"\n{'='*60}")
    print(f"QUANTITATIVE AGENT CYCLE INITIATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # 1. Initialize hardware interfaces
    router = Alpacarouter(paper=True)
    oracle = MarketOracle()

    # define the core trading universe
    try:
        with open('curated_universe.json', 'r') as f:
            universe = json.load(f)
        print(f">> Loaded {len(universe)} approved tickers from async compute node.")
    except Exception as e:
        print(f">> [CRITICAL] Failed to load curated universe: {e}")
        return

    # 2. The Shield: macroeconomic emergency brake
    print("\n[PHASE 1] Checking Macroeconomic Regime (CUSUM Filter)...")
    is_safe_to_trade = check_macro_safety()
    
    if not is_safe_to_trade:
        print(">> [WARNING] MACRO CRASH DETECTED. Market is unsafe.")
        router.liquidate_portfolio()
        print(">> Cycle Terminated. Standing by in cash.")
        return

    print(">> Macro Environment SAFE. Proceeding to Alpha Generation.")

    # 3. The Oracle & The Brain: data ingestion & signal generation
    print("\n[PHASE 2] Fetching Live Matrix and Scanning for Cointegration...")
    live_matrix = oracle.get_live_5m_bars(tickers=universe, lookback_days=7)
    
    if live_matrix.empty:
        print(">> Oracle returned no data. Cycle Terminated.")
        return

    # pass the RAM-only live data to the brain to find statistical divergences
    active_signals, spread_data = generate_signals(live_matrix) 
    
    if not active_signals:
        print(">> No statistically significant spreads found. Standing by.")
        return
        
    print(f">> Found {len(active_signals)} potential spread entries.")

    # 4. The Filter: meta-labeling & position sizing
    print("\n[PHASE 3] Meta-Labeling & Sizing (Loading XGBoost Model)...")
    
    # load the latest model parameters trained by the asynchronous compute node
    try:
        meta_labeler = xgb.XGBClassifier()
        meta_labeler.load_model('model/meta_labeler_v1.json')
    except Exception as e:
        print(f">> [CRITICAL] Failed to load Meta-Labeler JSON: {e}")
        return
    
    approved_spreads = {}
    total_kelly_fraction = 0.0
    
    for spread_name, spread_info in active_signals.items():
        # isolate the current features for this spread (last row of data)
        current_features = spread_data.loc[[spread_data.index[-1]]]
        
        # predict probability of the signal being profitable
        prob = meta_labeler.predict_proba(current_features)[:, 1][0]
        kelly_size = calculate_kelly_size(prob, payoff_ratio=2.0)
        
        if kelly_size > 0:
            approved_spreads[spread_name] = kelly_size
            total_kelly_fraction += kelly_size
            print(f">> {spread_name} Approved | Prob: {prob:.2f} | Kelly: {kelly_size:.2f}")
        else:
            print(f">> {spread_name} Rejected (Low Probability).")

    if not approved_spreads:
        print(">> All signals rejected by The Filter. Standing by.")
        return

    # 5. The Allocator: Hierarchical Risk Parity
    print("\n[PHASE 4] Structuring Portfolio Risk (HRP Clustering)...")
    account = router.trading_client.get_account()
    total_equity = float(account.portfolio_value)
    
    # calculate absolute dollar budget for this cycle
    total_risk_budget = total_equity * total_kelly_fraction
    print(f">> Total Equity: ${total_equity:,.2f} | Total Risk Budget: ${total_risk_budget:,.2f}")
    
    # HRP calculates exactly how to divide that budget based on covariance
    approved_returns = spread_data[list(approved_spreads.keys())]
    dollar_allocations = allocate_capital(approved_returns, total_risk_budget)

    # 6. The Executor: routing orders to Alpaca
    print("\n[PHASE 5] Translating Dollars to Fractional Shares...")
    for spread_name, dollar_amount in dollar_allocations.items():
        weights = active_signals[spread_name]['johansen_weights']
        target_pos = active_signals[spread_name]['target_position']
        
        leg_orders = router.calculate_leg_quantities(
            spread_allocation=dollar_amount,
            weights=weights,
            target_position=target_pos
        )
        
        if leg_orders:
            router.execute_spread(spread_name, leg_orders)

    print(f"\n{'='*60}")
    print("CYCLE COMPLETE. POSITIONS ROUTED TO ALPACA.")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("\n[SYSTEM BOOT] Quantitative Agent Online.")
    print("Press Ctrl+C to safely shutdown.")
    
    try:
        while True:
            current_hour = datetime.now().hour
            # Execute only during standard market hours (9 AM to 4 PM Eastern)
            # if 9 <= current_hour < 16:
            if True:  
                run_trading_cycle()
                print("\n[SLEEP] Waiting 5 minutes for the next bar to close...")
                time.sleep(300) 
            else:
                print(f"\n[{datetime.now().time()}] Market is closed. Sleeping for 1 hour...")
                time.sleep(3600)
                
    except KeyboardInterrupt:
        print("\n\n[SHUTDOWN] Manual override detected. Terminating agent safely.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")