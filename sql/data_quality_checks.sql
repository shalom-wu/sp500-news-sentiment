-- Checks that matter for a time-series sentiment analysis.

SELECT '01 daily rows' AS check_name, COUNT(*)::VARCHAR AS result
FROM daily_sentiment;

SELECT '02 headline rows' AS check_name, COUNT(*)::VARCHAR AS result
FROM headline_sentiment;

SELECT '03 duplicate daily dates' AS check_name,
       COUNT(*)::VARCHAR AS result
FROM (
    SELECT date
    FROM daily_sentiment
    GROUP BY 1
    HAVING COUNT(*) > 1
);

SELECT '04 missing next-day returns' AS check_name,
       COUNT(*)::VARCHAR AS result
FROM daily_sentiment
WHERE next_day_return IS NULL;

SELECT '05 valid next-trading-day pairs' AS check_name,
       ROUND(AVG(CASE WHEN next_day_ok THEN 1 ELSE 0 END) * 100, 1)::VARCHAR || '%' AS result
FROM daily_sentiment;

SELECT '06 headline sentiment score range' AS check_name,
       CASE WHEN COUNT(*) = 0 THEN 'pass' ELSE 'fail' END AS result
FROM headline_sentiment
WHERE sentiment_score < -1 OR sentiment_score > 1;

SELECT '07 source rows are real public headlines/prices' AS check_name,
       'documented in data-sources.md and data/data_manifest.md' AS result;
