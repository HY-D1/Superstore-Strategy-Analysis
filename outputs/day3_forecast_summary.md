# Day 3 Forecast + Customer Segmentation

## Forecast (Monthly Sales)
- Model: Linear trend baseline
- Backtest (last 6 months) MAPE: 29.85%

## Customer Concentration
- Total Sales (all time): $2,261,536.78
- Top 20% customers contribute ~48.29% of sales (revenue concentration).

## RFM Segments (Sales-only)
- Use Recency (days since last order), Frequency (unique orders), Monetary (total sales).
- See: `outputs/day3_rfm_segments.csv` and `outputs/day3_tables/rfm_segment_summary.csv`.

## Draft Strategy Ideas (Revenue-focused)
- **Retention:** target 'At Risk (High Value)' with reactivation offers and proactive outreach.
- **Upsell:** target 'Loyal' & 'Champions' with bundles in top sub-categories.
- **Ops:** use 'Ship Days' to identify slow delivery patterns and prioritize service improvements for high-value segments.

## Files Generated
- Forecast: `outputs/day3_forecast.csv`
- RFM: `outputs/day3_rfm_segments.csv`
- Chart: `outputs/day3_charts/monthly_sales_forecast.png`