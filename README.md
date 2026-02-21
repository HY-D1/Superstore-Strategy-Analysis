# Superstore Strategy Analysis

A portfolio-ready Strategy/Business Analyst project using the Superstore dataset.
Deliverables: cleaned Excel master file, insight pivots, forecast, and a Power BI dashboard.

## 🚀 Quick Start (One Command)

```bash
./start.sh
```

This runs the complete pipeline:
- **Day 1**: Data Cleaning + QA
- **Day 2**: Insights (charts + tables)
- **Day 3**: Forecast + RFM Segmentation
- **Day 4**: BI Export
- **Day 5**: Executive Summary + Slide Outline

### Run Individual Steps
```bash
./start.sh day1   # Data cleaning only
./start.sh day2   # Insights only
./start.sh day3   # Forecast + RFM only
./start.sh day4   # BI export only
./start.sh day5   # Story pack only
```

### 📊 Launch Interactive Dashboard
```bash
./start.sh dashboard        # Launch Streamlit dashboard
./start.sh full+dashboard   # Run pipeline then launch dashboard
```

The dashboard provides:
- 🏠 **Executive Summary** with key metrics and recommendations
- 📈 **Sales Insights** with interactive charts (trends, region, segment, category)
- 🔮 **Forecast** with 12-month projections
- 👥 **RFM Segmentation** with customer explorer
- 📦 **Data Explorer** with filtering and download

## Manual Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```