import duckdb
import pandas as pd
from the_research_node.backtesting.data import load_basket_bars

con = duckdb.connect()
con.execute("SET TimeZone = 'UTC'")
for name in ["01_profile.sql", "02_clean.sql", "03_hourly_bars.sql"]:
    con.execute(open(f"sql_reconciliation/queries/{name}").read())

# SQL side
sql_df = con.sql("SELECT * FROM hourly").df()
sql_df["bar_ts"] = pd.to_datetime(sql_df["bar_ts"], utc=True)

# pandas side (the original pipeline)
pd_df = load_basket_bars(["HYG", "TLT"])
pd_df.index.name = "bar_ts"
pd_df = pd_df.reset_index()

# compare
merged = pd_df.merge(sql_df, on="bar_ts", how="outer",
                    suffixes=("_pd", "_sql"), indicator=True)

only_pd  = (merged["_merge"] == "left_only").sum()   # rows where _merge == "left_only"
only_sql = (merged["_merge"] == "right_only").sum()   # rows where _merge == "right_only"
both = merged[merged["_merge"] == "both"]
bad = (
    ((both["HYG_pd"] - both["HYG_sql"]).abs() > 1e-9) 
    | ((both["TLT_pd"] - both["TLT_sql"]).abs() > 1e-9)
    ).sum()   # rows in both where abs(HYG_pd - HYG_sql) > 1e-9 OR same for TLT

print(f"{len(merged)} rows | only pandas: {only_pd} | only SQL: {only_sql} | mismatches: {bad}")