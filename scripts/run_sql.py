"""Run the DuckDB SQL layer and export Power BI-ready tables.

Run from the repo root:
    python scripts/run_sql.py
"""

from pathlib import Path
import os
import re

import duckdb


ROOT = Path(__file__).resolve().parents[1]
FILES = ["create_tables.sql", "data_quality_checks.sql", "kpi_views.sql", "analysis_queries.sql"]
SHOW = {"data_quality_checks.sql", "analysis_queries.sql"}
EXPORTS = {
    "v_daily_market_sentiment": "fact_daily_market_sentiment.csv",
    "v_sentiment_summary": "kpi_sentiment_summary.csv",
    "v_sentiment_by_year": "kpi_sentiment_by_year.csv",
    "v_sentiment_tercile_forward_returns": "kpi_sentiment_tercile_forward_returns.csv",
    "v_news_volume_by_month": "kpi_news_volume_by_month.csv",
}


def main() -> None:
    os.chdir(ROOT)
    con = duckdb.connect()
    for name in FILES:
        print(f"\n=== {name} ===")
        sql = (ROOT / "sql" / name).read_text(encoding="utf-8")
        for stmt in [s.strip() for s in re.split(r";\s*(?:\n|$)", sql) if s.strip()]:
            result = con.execute(stmt)
            body = re.sub(r"^\s*(--[^\n]*\n)*", "", stmt).lstrip().upper()
            if name in SHOW and body.startswith(("SELECT", "WITH")):
                print(result.df().to_string(index=False, max_rows=30))
                print()

    out = ROOT / "data" / "powerbi"
    out.mkdir(parents=True, exist_ok=True)
    print("\n=== exports for Power BI ===")
    for view, fname in EXPORTS.items():
        target = out / fname
        if target.exists():
            try:
                target.unlink()
            except PermissionError:
                n = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
                print(f"  {fname}: kept existing locked file; expected {n:,} rows")
                continue
        con.execute(
            f"COPY (SELECT * FROM {view}) TO '{target.as_posix()}' "
            "(HEADER, DELIMITER ',')"
        )
        n = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
        print(f"  {fname}: {n:,} rows")


if __name__ == "__main__":
    main()
