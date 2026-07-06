# Manual Build Instructions

1. Run `python scripts/run_sql.py`.
2. Open Power BI Desktop.
3. Load all CSVs from `data/powerbi/`.
4. Create the measures in `dax_measures.md`.
5. Build three pages:
   - Executive KPI: cards for trading days, headlines, next-day correlation, next-day up rate; line chart for headline volume; bar chart by sentiment bucket.
   - Diagnostic Analysis: year trend chart and monthly news volume chart; slicers for year and valid next-day pair.
   - Decision Support: table with sentiment buckets and a text box explaining the weak predictive signal.
6. Add footer text: `Source: Kaggle S&P 500 headlines dataset; FinBERT sentiment; date-only headline caveat applies.`
7. Save as `power-bi/sp500_news_sentiment.pbix`.

The included mockups are layout guides, not actual Power BI Desktop screenshots.
