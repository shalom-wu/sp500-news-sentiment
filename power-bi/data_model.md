# Data Model

Load these files from `data/powerbi/`:

| Table | File | Grain |
|---|---|---|
| `fact_daily_market_sentiment` | `fact_daily_market_sentiment.csv` | One row per observed trading date. |
| `kpi_sentiment_summary` | `kpi_sentiment_summary.csv` | One-row summary. |
| `kpi_sentiment_by_year` | `kpi_sentiment_by_year.csv` | One row per year. |
| `kpi_sentiment_tercile_forward_returns` | `kpi_sentiment_tercile_forward_returns.csv` | One row per sentiment bucket. |
| `kpi_news_volume_by_month` | `kpi_news_volume_by_month.csv` | One row per month. |

Relationships are not required for a simple report because the exported tables are already at dashboard grain. Use `fact_daily_market_sentiment[date]` for date slicers if you add a date table.
