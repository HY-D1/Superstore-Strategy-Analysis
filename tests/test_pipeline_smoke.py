from __future__ import annotations

import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def test_end_to_end_smoke() -> None:
    # Ensure sample data exists
    if not Path("data_raw/sample_train.csv").exists():
        run(["python", "scripts/generate_sample_data.py"])

    # Day 1
    run(["python", "src/day1_clean.py", "--input", "data_raw/sample_train.csv", "--output", "data_clean/Superstore_Cleaned.xlsx"])
    assert Path("data_clean/Superstore_Cleaned.xlsx").exists()

    # Day 2
    run(["python", "src/day2_insights.py", "--input", "data_clean/Superstore_Cleaned.xlsx"])
    assert Path("outputs/day2_insights.md").exists()

    # Day 3
    run(["python", "src/day3_forecast_rfm.py", "--input", "data_clean/Superstore_Cleaned.xlsx", "--horizon", "12"])
    assert Path("outputs/day3_forecast_summary.md").exists()
    assert Path("outputs/day3_charts/monthly_sales_forecast.png").exists()

    # Day 4
    run(["python", "src/day4_export_bi.py", "--master_xlsx", "data_clean/Superstore_Cleaned.xlsx",
         "--rfm_csv", "outputs/day3_rfm_segments.csv", "--out_csv", "outputs/bi/superstore_bi.csv"])
    assert Path("outputs/bi/superstore_bi.csv").exists()

    # Day 5
    run(["python", "src/day5_story_pack.py"])
    assert Path("outputs/day5_executive_summary.md").exists()
    assert Path("docs/day5_slide_outline.md").exists()
