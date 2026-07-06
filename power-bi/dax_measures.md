# DAX Measures

Create these on `fact_daily_market_sentiment`:

```DAX
Trading Days = COUNTROWS(fact_daily_market_sentiment)

Scored Headlines = SUM(fact_daily_market_sentiment[headline_count])

Avg Sentiment = AVERAGE(fact_daily_market_sentiment[sentiment_mean])

Avg Next-Day Return = AVERAGE(fact_daily_market_sentiment[next_day_return])

Next-Day Up Rate =
DIVIDE(
    COUNTROWS(FILTER(fact_daily_market_sentiment, fact_daily_market_sentiment[next_day_return] > 0)),
    COUNTROWS(FILTER(fact_daily_market_sentiment, NOT ISBLANK(fact_daily_market_sentiment[next_day_return])))
)
```

The Pearson correlations are precomputed in SQL in `kpi_sentiment_summary.csv`; use those as cards rather than re-implementing correlation in DAX.
