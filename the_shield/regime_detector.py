import os
import numpy as np
import pandas as pd
import yfinance as yf

def fetch_macro_data(tickers: list, start_date: str, end_date: str, output_path: str) -> pd.DataFrame:
    # pepe: download the raw macroeconomic indicators
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    data = data.ffill().dropna()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path)
    return data

class RegimeCUSUM:
    """
    Symmetric CUSUM Filter acting as the Macroeconomic Emergency Brake.
    Locked to 10.0x multiplier on log(VIX) daily differences.
    """
    def __init__(self, threshold_multi: float = 10.0):
        self.threshold_multi = threshold_multi
        self.breaks = []
        self.current_regime = "NORMAL" # Defaults to normal until a break is detected

    def fit_predict(self, series: pd.Series) -> pd.Series:
        diffs = series.diff().dropna()
        h = diffs.std() * self.threshold_multi
        
        s_pos = 0.0
        s_neg = 0.0
        events = pd.Series(0, index=series.index, name='CUSUM_Breaks')
        
        for i in range(1, len(series)):
            diff_val = series.iloc[i] - series.iloc[i-1]
            s_pos = max(0, s_pos + diff_val)
            s_neg = min(0, s_neg + diff_val)
            
            # pepe: VIX spikes upward (Macro Crash / Contagion)
            if s_pos > h:
                s_pos, s_neg = 0.0, 0.0
                events.iloc[i] = 1
                self.breaks.append((series.index[i], 'CRASH_REGIME'))
                self.current_regime = "CRASH_REGIME"
                
            # pepe: VIX drops downward (Volatility Suppression / Safe to Trade)
            elif s_neg < -h:
                s_pos, s_neg = 0.0, 0.0
                events.iloc[i] = -1
                self.breaks.append((series.index[i], 'NORMAL_REGIME'))
                self.current_regime = "NORMAL"
                
        return events

def check_macro_safety() -> bool:
    """
    The main API function for the Master Orchestrator. 
    Returns True if the market is safe to trade, False if the portfolio should be liquidated.
    """
    raw_path = '/app/data/raw_macro_data.csv'
    # In live execution, we only need a short lookback window to check the current state
    df_raw = fetch_macro_data(['^VIX'], start_date="2024-01-01", end_date=pd.Timestamp.today().strftime('%Y-%m-%d'), output_path=raw_path)
    
    # 1. Stationary Transformation (d=0.0)
    log_vix = np.log(df_raw['^VIX'])
    
    # 2. Arm the Shield
    shield = RegimeCUSUM(threshold_multi=10.0)
    shield.fit_predict(log_vix)
    
    # 3. Output the binary safety state
    is_safe = shield.current_regime == "NORMAL"
    return is_safe

if __name__ == "__main__":
    print("\n=== SYSTEM CHECK: THE SHIELD ===")
    
    # Run the historical diagnostic
    shield_test = RegimeCUSUM(threshold_multi=10.0)
    df = fetch_macro_data(['^VIX'], "2012-01-01", "2026-03-26", '/app/data/raw_macro_data.csv')
    shield_test.fit_predict(np.log(df['^VIX']))
    
    print(f"\nTotal Historical Emergency Overrides: {len(shield_test.breaks)}")
    print("\nMost Recent Macro Regime Shifts:")
    for date, event in shield_test.breaks[-5:]:
        print(f"[{date.date()}] -> {event}")
        
    print(f"\nCURRENT MARKET STATE SAFE TO TRADE? -> {check_macro_safety()}")