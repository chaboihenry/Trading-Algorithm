import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# set professional plotting style
sns.set_theme(style="darkgrid", palette="viridis")

def fetch_historical_data(tickers, start_date, end_date):
    print("Fetching historical data...")
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    return data.ffill().dropna()

def simulate_portfolio(df_close):
    print("Calculating Cointegrated Spreads...")
    
    # 1. define the pristine baskets
    baskets = {
        'Financials (V/MA)': {'V': 1.0, 'MA': -0.5412},
        'Tech (AAPL/MSFT)': {'AAPL': 1.0, 'MSFT': -0.8921},
        'Semis (NVDA/AMD)': {'NVDA': 1.0, 'AMD': -1.205}
    }
    
    portfolio_returns = pd.Series(0.0, index=df_close.index)
    
    for name, weights in baskets.items():
        # calculate Spread Price
        spread_price = pd.Series(0.0, index=df_close.index)
        for ticker, w in weights.items():
            spread_price += df_close[ticker] * w
            
        # calculate Rolling Z-Score (Using roughly 14 days for daily data)
        rolling_mean = spread_price.rolling(window=14).mean()
        rolling_std = spread_price.rolling(window=14).std()
        z_score = (spread_price - rolling_mean) / rolling_std
        
        # generate Signals (+1, -1, 0)
        signals = pd.Series(0, index=df_close.index)
        signals[z_score < -2.0] = 1
        signals[z_score > 2.0] = -1
        # exit when reverting to mean
        signals[(z_score >= 0) & (signals.shift(1) == 1)] = 0
        signals[(z_score <= 0) & (signals.shift(1) == -1)] = 0
        signals = signals.ffill().fillna(0)
        
        # shift signals by 1 to prevent look-ahead bias (we trade the NEXT day's return)
        signals = signals.shift(1).fillna(0)
        
        # calculate Spread Log Returns
        spread_returns = np.log(spread_price / spread_price.shift(1)).dropna()
        
        # strategy Returns for this specific spread
        strategy_returns = signals * spread_returns
        
        # add to total portfolio (Simulating HRP Equal-Weighting for the backtest)
        weight_per_spread = 1.0 / len(baskets)
        portfolio_returns = portfolio_returns.add(strategy_returns * weight_per_spread, fill_value=0)

    return portfolio_returns.dropna()

def generate_tearsheet(returns):
    print("Generating Institutional Tearsheet...")
    
    # cumulative Returns
    cumulative_returns = (1 + returns).cumprod()
    
    # metrics
    trading_days = 252
    annualized_return = returns.mean() * trading_days
    annualized_volatility = returns.std() * np.sqrt(trading_days)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('Quantitative Agent: Statistical Arbitrage Portfolio', fontsize=18, fontweight='bold', color='white')
    fig.patch.set_facecolor('#1e1e1e')
    
    # equity Curve
    ax1.set_facecolor('#1e1e1e')
    ax1.plot(cumulative_returns.index, cumulative_returns, color='#00ffcc', linewidth=2)
    ax1.set_title(f'Cumulative Portfolio Equity (Sharpe: {sharpe_ratio:.2f})', color='white')
    ax1.set_ylabel('Growth of $1', color='white')
    ax1.tick_params(colors='white')
    ax1.grid(color='#333333')
    
    # Text Box with Stats
    stats_text = (
        f"Annualized Return: {annualized_return*100:.2f}%\n"
        f"Annualized Vol: {annualized_volatility*100:.2f}%\n"
        f"Max Drawdown: {max_drawdown*100:.2f}%\n"
        f"Sharpe Ratio: {sharpe_ratio:.2f}"
    )
    ax1.text(0.02, 0.85, stats_text, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.6, edgecolor='#00ffcc'), color='white')

    # Drawdown Curve
    ax2.set_facecolor('#1e1e1e')
    ax2.fill_between(drawdown.index, drawdown, 0, color='#ff3333', alpha=0.5)
    ax2.plot(drawdown.index, drawdown, color='#ff3333', linewidth=1)
    ax2.set_title('Portfolio Drawdown', color='white')
    ax2.set_ylabel('Drawdown %', color='white')
    ax2.tick_params(colors='white')
    ax2.grid(color='#333333')

    plt.tight_layout()
    
    # save the graph for GitHub
    plt.savefig('backtest_tearsheet.png', facecolor=fig.get_facecolor(), edgecolor='none')
    print("\n[SUCCESS] backtest_tearsheet.png saved to root directory!")
    plt.show()

if __name__ == "__main__":
    universe = ['V', 'MA', 'AAPL', 'MSFT', 'NVDA', 'AMD']
    # 3-Year Backtest Window
    historical_df = fetch_historical_data(universe, "2021-01-01", "2024-01-01")
    
    port_returns = simulate_portfolio(historical_df)
    generate_tearsheet(port_returns)