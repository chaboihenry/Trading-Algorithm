"""Pull CRSP cfacpr split adjustment factors for the universe.

CRSP's cfacpr is the cumulative factor to adjust price. To get the
split-adjusted equivalent of a raw historical price in current units:
    adjusted = raw_price / cfacpr_at_that_date

Saves output to ~/quant_data/split_factors.parquet
"""

import os
import wrds
import pandas as pd

WRDS_USERNAME = "henryvianna"
UNIVERSE_PATH = "universe.txt"
OUTPUT_PATH = os.path.expanduser("~/quant_data/split_factors.parquet")
START_DATE = "2021-01-01"
END_DATE = "2026-12-31"


def load_universe():
    with open(UNIVERSE_PATH, "r") as f:
        return [line.strip() for line in f if line.strip()]


def fetch_split_factors(db, tickers):
    # Query CRSP dsf joined with stocknames for accurate ticker resolution
    tickers_sql = ", ".join(f"'{t}'" for t in tickers)
    query = f"""
        SELECT sn.ticker, dsf.date, dsf.cfacpr
        FROM crsp.dsf dsf
        INNER JOIN crsp.stocknames sn ON dsf.permno = sn.permno
        WHERE sn.ticker IN ({tickers_sql})
          AND dsf.date BETWEEN '{START_DATE}' AND '{END_DATE}'
          AND dsf.date BETWEEN sn.namedt AND COALESCE(sn.nameenddt, '9999-12-31')
        ORDER BY sn.ticker, dsf.date
    """
    df = db.raw_sql(query)
    df['date'] = pd.to_datetime(df['date'])
    return df


def main():
    universe = load_universe()
    print(f"Loaded {len(universe)} tickers from {UNIVERSE_PATH}")

    print(f"Connecting to WRDS as {WRDS_USERNAME}...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)
    print("Connected.")

    print(f"Fetching cfacpr from CRSP {START_DATE} to {END_DATE}...")
    df = fetch_split_factors(db, universe)
    db.close()

    print(f"Fetched {len(df):,} rows for {df['ticker'].nunique()} tickers.")

    # Identify tickers with actual splits (where cfacpr varies)
    by_ticker = df.groupby('ticker')['cfacpr'].agg(['min', 'max', 'nunique'])
    affected = by_ticker[by_ticker['nunique'] > 1].copy()
    affected['split_ratio'] = affected['max'] / affected['min']
    print(f"\nTickers with splits in this window ({len(affected)}):")
    for ticker, row in affected.iterrows():
        print(f"  {ticker:8s}: cfacpr {row['min']:.4f} -> {row['max']:.4f} "
              f"(ratio {row['split_ratio']:.2f})")

    # Tickers in universe with NO CRSP data — possible naming mismatches
    fetched_tickers = set(df['ticker'].unique())
    universe_set = set(universe)
    missing = universe_set - fetched_tickers
    if missing:
        print(f"\n[WARNING] {len(missing)} universe tickers had no CRSP data:")
        print(f"  {sorted(missing)}")
        print("  These may be ETFs (no CRSP coverage) or ticker mismatches.")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, compression='zstd')
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()