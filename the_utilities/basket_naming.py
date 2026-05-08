def canonical_basket_key(tickers) -> str:
    return "_".join(sorted(tickers)) + "_Spread"


def canonical_ticker_set(spread_name: str) -> frozenset:
    name = spread_name.replace("_Spread", "")
    return frozenset(name.split("_"))


def baskets_equivalent(name_a: str, name_b: str) -> bool:
    return canonical_ticker_set(name_a) == canonical_ticker_set(name_b)