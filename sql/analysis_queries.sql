-- Business-facing checks for the central claim: sentiment is weak as a next-day predictor.

SELECT *
FROM v_sentiment_summary;

SELECT *
FROM v_sentiment_tercile_forward_returns;

SELECT *
FROM v_sentiment_by_year
ORDER BY year;

SELECT
    'High minus low sentiment next-day return' AS metric,
    ROUND(
        (SELECT avg_next_day_return_pct FROM v_sentiment_tercile_forward_returns WHERE sentiment_bucket = 'High sentiment')
        -
        (SELECT avg_next_day_return_pct FROM v_sentiment_tercile_forward_returns WHERE sentiment_bucket = 'Low sentiment'),
        3
    ) AS value_pct_points;
