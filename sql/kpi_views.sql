-- SQL reference outputs for the Power BI handoff.

CREATE OR REPLACE VIEW v_sentiment_summary AS
SELECT
    COUNT(*) AS trading_days,
    MIN(date) AS first_date,
    MAX(date) AS last_date,
    SUM(headline_count) AS scored_headlines,
    ROUND(AVG(headline_count), 2) AS avg_headlines_per_day,
    ROUND(CORR(sentiment_mean, next_day_return), 4) AS corr_sentiment_next_return,
    ROUND(CORR(sentiment_mean, same_day_return), 4) AS corr_sentiment_same_day_return,
    ROUND(AVG(CASE WHEN next_day_return > 0 THEN 1 ELSE 0 END) * 100, 1) AS next_day_up_rate_pct
FROM daily_sentiment
WHERE next_day_return IS NOT NULL;

CREATE OR REPLACE VIEW v_sentiment_by_year AS
SELECT
    EXTRACT(year FROM date)::INTEGER AS year,
    COUNT(*) AS trading_days,
    SUM(headline_count) AS headlines,
    ROUND(AVG(sentiment_mean), 4) AS avg_sentiment,
    ROUND(AVG(next_day_return) * 100, 3) AS avg_next_day_return_pct,
    ROUND(AVG(CASE WHEN next_day_return > 0 THEN 1 ELSE 0 END) * 100, 1) AS next_day_up_rate_pct
FROM daily_sentiment
WHERE next_day_return IS NOT NULL
GROUP BY 1
ORDER BY 1;

CREATE OR REPLACE VIEW v_sentiment_tercile_forward_returns AS
WITH scored AS (
    SELECT
        *,
        NTILE(3) OVER (ORDER BY sentiment_mean) AS sentiment_tercile
    FROM daily_sentiment
    WHERE next_day_return IS NOT NULL
)
SELECT
    CASE sentiment_tercile
        WHEN 1 THEN 'Low sentiment'
        WHEN 2 THEN 'Middle sentiment'
        ELSE 'High sentiment'
    END AS sentiment_bucket,
    COUNT(*) AS trading_days,
    ROUND(AVG(sentiment_mean), 4) AS avg_sentiment,
    ROUND(AVG(next_day_return) * 100, 3) AS avg_next_day_return_pct,
    ROUND(AVG(CASE WHEN next_day_return > 0 THEN 1 ELSE 0 END) * 100, 1) AS next_day_up_rate_pct
FROM scored
GROUP BY 1, sentiment_tercile
ORDER BY sentiment_tercile;

CREATE OR REPLACE VIEW v_news_volume_by_month AS
SELECT
    DATE_TRUNC('month', date)::DATE AS month_start,
    COUNT(*) AS trading_days,
    SUM(headline_count) AS headlines,
    ROUND(AVG(sentiment_mean), 4) AS avg_sentiment,
    ROUND(AVG(next_day_return) * 100, 3) AS avg_next_day_return_pct
FROM daily_sentiment
GROUP BY 1
ORDER BY 1;
