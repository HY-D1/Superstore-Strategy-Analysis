# Superstore Strategy Analysis (Sales, Ops & Retention)

Portfolio project for Strategy / Business Analyst internships.  
Built an end-to-end analysis workflow (cleaning → insights → forecasting → customer segmentation → dashboard-ready outputs) using Excel + Python + Jupyter + Docker.

## Features
- Cleaned and standardized **9,800** transactions; created derived fields (Year/Month/Year-Month, Ship Days) and a QA report.
- Produced sales insights across **Region / Segment / Category / Sub-Category**, plus customer concentration and retention proxy metrics.
- Built a baseline **monthly sales forecast** and an **RFM customer segmentation** (Champions / Loyal / At-Risk) to drive retention strategy.

## Repo contents
- `src/day1_clean.py` → clean raw CSV → `data_clean/Superstore_Cleaned.xlsx` + QA report
- `src/day2_insights.py` → generate tables/charts + `outputs/day2_insights.md`
- `src/day3_forecast_rfm.py` → baseline forecast + RFM segments
- `notebooks/` → Jupyter notebooks for exploration and visuals
- `outputs/` → generated charts/tables/reports (tracked for portfolio)

## Quickstart (local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Day 1: Clean + QA
```bash
python src/day1_clean.py --input data_raw/train.csv --output data_clean/Superstore_Cleaned.xlsx
```

### Day 2: Insights (charts + report)
```bash
python src/day2_insights.py --input data_clean/Superstore_Cleaned.xlsx
```

### Day 3: Forecast + RFM segmentation
```bash
python src/day3_forecast_rfm.py --input data_clean/Superstore_Cleaned.xlsx --horizon 12
```

## Jupyter
```bash
jupyter lab
```

## Docker
```bash
docker build -t superstore .
docker run --rm -v "$PWD":/app superstore
```

### Data source
Superstore Sales dataset from Kaggle (see `data_raw/`).
If you reuse this project, verify dataset licensing/redistribution rules.