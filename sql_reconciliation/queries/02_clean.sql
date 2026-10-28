CREATE OR REPLACE VIEW cleaned AS
    SELECT ticker, ts, price
    FROM (
        SELECT
            ticker,
            ts,
            price,
            timezone('America/New_York', ts) AS ts_et,
        FROM ticks
    )
    WHERE price IS NOT NULL
        AND isodow(ts_et) BETWEEN 1 AND 5
        AND hour(ts_et) * 60 + minute(ts_et) BETWEEN 570 AND 960;