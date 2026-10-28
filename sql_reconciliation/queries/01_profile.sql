CREATE OR REPLACE VIEW ticks AS
    SELECT 
        split_part(filename, '/', 6) AS ticker,
        CAST(timestamp AS TIMESTAMPTZ) AS ts,
        price
    FROM read_parquet(
    'C:/Users/henry/quant_data/tick_data_storage/*/parquet/training_data/*.parquet',
    filename = true
    );
    SELECT
        ticker,
        COUNT(*) AS n_rows,
        MIN(ts) AS first_ts,
        MAX(ts) AS last_ts,
        COUNT(*) - COUNT(price) AS null_prices,
        COUNT(*) - COUNT(DISTINCT (ts, price)) AS dup_rows  
    FROM ticks
    GROUP BY ticker;