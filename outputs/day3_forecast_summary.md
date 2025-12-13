# Day 3 Forecast + Customer Segmentation

## Forecast (Monthly Sales)
- Models compared (rolling backtest): linear, seasonal_naive, seasonal_avg, seasonal_regression
- Selected model: **seasonal_regression**
- Rolling backtest: horizon=6 months, min_train=24 months, splits=19
- Avg MAPE (selected): 20.63%
- Next 12 months forecast total: $736,715.27 (avg/month: $61,392.94)
- See: `outputs/day3_tables/model_backtest_comparison.csv`

## Customer Concentration
- Total Sales (all time): $2,261,536.78
- Top 20% customers contribute ~48.29% of sales (revenue concentration).

## RFM Segments (Sales-only)
- Recency = days since last order; Frequency = unique orders; Monetary = total sales.
- See: `outputs/day3_rfm_segments.csv` and `outputs/day3_tables/rfm_segment_summary.csv`.

## Draft Strategy Ideas (Revenue-focused)
- **Retention:** target 'At Risk (High Value)' with reactivation outreach.
- **Upsell:** target 'Loyal' & 'Champions' with bundles in top sub-categories.
- **Ops:** prioritize faster shipping for high-value segments where Ship Days are high.

## Files Generated
- Forecast: `outputs/day3_forecast.csv`
- Backtest: `outputs/day3_tables/model_backtest_comparison.csv`
- RFM: `outputs/day3_rfm_segments.csv`
- Chart: `outputs/day3_charts/monthly_sales_forecast.png`