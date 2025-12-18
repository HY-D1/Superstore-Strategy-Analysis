# Day 3 Forecast + Customer Segmentation

## Forecast (Monthly Sales)
- Selected model: **Trend+MonthSeasonality** (lowest RollingMAPE_h3)
- Rolling backtest MAPE (3-month horizon): 20.97% over 22 tests

## Customer Concentration
- Total Sales (all time): $2,261,536.78
- Top 20% customers contribute ~48.29% of sales (revenue concentration).

## RFM Segments (Sales-only)
- Recency = days since last order (lower is better)
- Frequency = unique orders (if Order ID exists), else row count
- Monetary = total sales

## Files Generated
- Monthly series: `outputs/day3_tables/monthly_sales.csv`
- Model comparison: `outputs/day3_tables/model_backtest_comparison.csv`
- Forecast: `outputs/day3_forecast.csv`
- RFM: `outputs/day3_rfm_segments.csv`
- Segment summary: `outputs/day3_tables/rfm_segment_summary.csv`
- Chart: `outputs/day3_charts/monthly_sales_forecast.png`