import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from the_brain.cointegration_arbitrage.cointegration_tests import test_johansen_cointegration

# a subset of stocks to test the clustering logic
tickers = [
    'AAPL', 'ABBV', 'AFRM', 'AI', 'AMAT', 'AMC', 'AMD', 'AMGN', 'AMZN', 
    'AVGO', 'AXP', 'BA', 'BAC', 'BLK', 'C', 'CAT', 'CMCSA', 'COIN', 'COP', 
    'COST', 'CVS', 'CVX', 'DE', 'DIA', 'DIS', 'DKNG', 'DUK', 'EEM', 'EOG', 
    'F', 'FDX', 'GE', 'GILD', 'GM', 'GME', 'GOOG', 'GOOGL', 'GS', 'HD', 
    'HON', 'HOOD', 'HYG', 'INTC', 'IWM', 'JNJ', 'JPM', 'KO', 'LCID', 'LLY', 
    'LMT', 'LOW', 'LRCX', 'LYFT', 'MA', 'MARA', 'MCD', 'META', 'MRK', 'MS', 
    'MSFT', 'MSTR', 'MU', 'NEE', 'NFLX', 'NIO', 'NKE', 'NVDA', 'OXY', 'PEP', 
    'PFE', 'PG', 'PLTR', 'QCOM', 'QQQ', 'RIOT', 'RIVN', 'ROKU', 'RTX', 
    'SBUX', 'SHOP', 'SLB', 'SMH', 'SO', 'SOFI', 'SPY', 'SQQQ', 'T', 'TGT', 
    'TLT', 'TQQQ', 'TSLA', 'TSM', 'TXN', 'UBER', 'UNH', 'UPS', 'UPST', 
    'UVXY', 'V', 'VXX', 'VZ', 'WFC', 'WMT', 'XLB', 'XLE', 'XLF', 'XLI', 
    'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XOM'
]

def discover_clusters(ticker_list):
    print("1. Downloading daily data for 10 stocks...")
    data = yf.download(ticker_list, period="2y", interval="1d")['Close']
    data = data.dropna(axis=1)

    print("2. Calculating and standardizing daily returns...")
    returns = data.pct_change().dropna()
    returns_t = returns.T

    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns_t)

    print("3. Running PCA on CPU to extract core economic behaviors...")
    pca = PCA(n_components=5)
    pca_components = pca.fit_transform(scaled_returns)

    print("4. Unleashing DBSCAN to find dense economic neighborhoods...")
    # eps controls how tight the leash must be. min_samples=2 requires pairs.
    dbscan = DBSCAN(eps=1.2, min_samples=2)
    clusters = dbscan.fit_predict(pca_components)

    results = pd.DataFrame({
        'Ticker': returns_t.index,
        'Cluster': clusters
    })

    print("\n=== RAW CLUSTERING RESULTS ===")
    # temporarily force pandas to print every single row
    with pd.option_context('display.max_rows', None):
        print(results.sort_values('Cluster'))
    print("==============================\n")

    print("\n=== POTENTIAL PAIRS DISCOVERED ===")
    # filter out the noise (-1) to only show the stocks that were grouped
    valid_clusters = results[results['Cluster'] != -1].sort_values('Cluster')
    
    if len(valid_clusters) == 0:
        print("No pairs found! The bouncer was too strict. Increase 'eps'.")
    else:
        print(valid_clusters)
    print("==================================\n")

    print("\n=== GPU-ACCELERATED JOHANSEN COINTEGRATION VERIFICATION ===")
    # group the valid tickers by their cluster id
    grouped_clusters = valid_clusters.groupby('Cluster')['Ticker'].apply(list)
    
    confirmed_baskets = []

    for cluster_id, cluster_tickers in grouped_clusters.items():
        print(f"\nEvaluating Cluster {cluster_id} containing: {cluster_tickers}")
        
        # dynamically construct the paths for every stock in the cluster simultaneously
        paths = [f"/app/data/tick data storage/{ticker}/parquet/ticks.parquet" for ticker in cluster_tickers]

        # feed entire system into the var model and unpack the new metrics
        is_cointegrated, half_life, weights = test_johansen_cointegration(paths)

        if is_cointegrated:
            # tradability filter
            valid_half_life = (0.01 <= half_life <= 15.0)
            valid_weights = all(0.1 <= abs(w) <= 10.0 for w in weights.values())
            if valid_half_life and valid_weights:
                print("Passed: Portfolio is mathematically tradeable")
                confirmed_baskets.append({
                    'basket': cluster_tickers,
                    'half_life': half_life,
                    'weights': weights
                })
            else:
                print("Rejected: Filter Triggered. Degenerate weights or extreme half-life detected.")
        
    print("\n=== FINAL SURVIVORS (TRADABLE PORTFOLIOS) ===")
    for idx, data in enumerate(confirmed_baskets):
        print(f"\nBasket {idx}: {data['basket']}")
        print(f"Mean-Reversion Half-Life: {data['half_life']:.2f} Trading Days")
        print("Hedge Ratios (Normalized):")
        for ticker, weight in data['weights'].items():
            print(f"  {ticker}: {weight:.4f}")
    print("===============================================\n")

if __name__ == "__main__":
    discover_clusters(tickers)
