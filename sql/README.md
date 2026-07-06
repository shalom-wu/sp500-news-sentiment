# SQL Layer - Market/Sentiment Validation

This folder uses DuckDB to validate and aggregate the included S&P 500 headline sentiment outputs. The SQL layer is intentionally modest: the project is not claiming a trading edge, so SQL focuses on reproducible checks, correlation summaries, tercile cuts, and Power BI-ready trend tables.

Run it from the project root:

```bash
python scripts/run_sql.py
```

Files run in order:

| File | Purpose |
|---|---|
| `create_tables.sql` | Creates local views over `data/processed/daily_sentiment.csv` and `headline_sentiment.csv`. |
| `data_quality_checks.sql` | Checks daily grain, missing forward returns, headline score ranges, and next-trading-day coverage. |
| `kpi_views.sql` | Defines the summary, year, tercile, and monthly views used by Power BI. |
| `analysis_queries.sql` | Recomputes the headline diagnostic cuts for review. |

Exports written to `data/powerbi/`:

- `fact_daily_market_sentiment.csv`
- `kpi_sentiment_summary.csv`
- `kpi_sentiment_by_year.csv`
- `kpi_sentiment_tercile_forward_returns.csv`
- `kpi_news_volume_by_month.csv`
