"""Verify every ticker in the fixed-income basket is tradable and shortable
on Alpaca before building a backtest that assumes we can trade them."""
import os
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST

BASKET = ["AGG", "BND", "EMB", "HYG", "HYLB", "IEF",
          "JNK", "LQD", "SHY", "TIP", "TLT"]


def main():
    load_dotenv()
    api = REST(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_API_SECRET"),
        os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets"),
    )
    header = f"{'Ticker':8s} {'tradable':9s} {'shortable':10s} {'easy_borrow':12s} fractionable"
    print(header)
    for t in BASKET:
        try:
            a = api.get_asset(t)
            print(f"{t:8s} {str(a.tradable):9s} {str(a.shortable):10s} "
                  f"{str(a.easy_to_borrow):12s} {a.fractionable}")
        except Exception as e:
            print(f"{t:8s} ERROR: {e}")


if __name__ == "__main__":
    main()
