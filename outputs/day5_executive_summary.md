# Executive Summary — Superstore Strategy Analysis

**Coverage:** 2015-01 to 2018-12

## Objective
- Understand sales drivers, customer concentration, and shipping performance; create a forecast baseline and retention-focused segmentation.

## Key Findings (Sales-only dataset)
- **Total sales:** $2,261,537
- **Top region:** West (31.4% of sales)
- **Top segment:** Consumer (50.8% of sales)
- **Top category:** Technology (36.6% of sales)
- **Top sub-category:** Phones ($327,782)
- **Shipping speed:** best = Same Day (0.0 days), worst = Standard Class (5.0 days)
- **Customer concentration:** Top 20% customers contribute ~48.3% of sales

## Forecast (Next 12 months)
- Model: Trend+MonthSeasonality | RollingMAPE_h3 = 21.0%
- Forecast total (next 12 months): $832,796
- vs last 12 months actual: 15.3%

## Recommendations (Actionable)
- **Retention:** prioritize At-Risk high-value customers with reactivation offers and outreach.
- **Growth:** upsell/bundle offers for Loyal + Champions in top-performing sub-categories.
- **Ops:** reduce Ship Days for high-value segments (set shipping SLA targets by Ship Mode).

## Evidence (Repo outputs)
- `outputs/day2_insights.md` (insights memo)
- `outputs/day2_tables/` and `outputs/day2_charts/` (breakdowns + visuals)
- `outputs/day3_forecast.csv` + `outputs/day3_charts/monthly_sales_forecast.png`
- `outputs/day3_rfm_segments.csv` + `outputs/day3_tables/rfm_segment_summary.csv`
