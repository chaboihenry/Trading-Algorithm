CREATE OR REPLACE VIEW hourly AS
    WITH bars AS (
        SELECT 
            ticker,
            date_trunc('hour', ts) AS bar_ts,
            arg_max(price, ts) AS raw_close
        FROM cleaned
        GROUP BY ticker, bar_ts
    ),
    factors AS (
        SELECT 
            ticker,
            CAST(date AS DATE) AS factor_date,
            CAST(cfacpr AS DOUBLE) AS cfacpr
            FROM read_parquet('C:/Users/henry/quant_data/split_factors.parquet')
    ),
    adjusted AS (
        SELECT 
            b.ticker,
            b.bar_ts,
            b.raw_close / COALESCE(f.cfacpr, 1.0) AS close
        FROM bars b
        ASOF LEFT JOIN factors f
            ON b.ticker = f.ticker
            AND CAST(timezone('America/New_York', b.bar_ts) AS DATE) >= f.factor_date
    )
    SELECT 
        bar_ts,
        MAX(close) FILTER (WHERE ticker = 'HYG') AS HYG,
        MAX(close) FILTER (WHERE ticker = 'TLT') AS TLT
    FROM adjusted
    GROUP BY bar_ts
    HAVING (HYG IS NOT NULL) AND (TLT IS NOT NULL)
    ORDER BY bar_ts;