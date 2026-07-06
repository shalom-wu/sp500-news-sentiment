# Refresh Instructions

```bash
python scripts/run_sql.py
```

Then open the manually built `.pbix`, check that the CSV sources point to `data/powerbi/`, and use Home -> Refresh.

If the sentiment pipeline is rerun from scratch, run the notebooks in order first, then rerun the SQL exporter.
