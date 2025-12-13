from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def money(x: float) -> str:
    return f"${x:,.2f}"


def pct(x: float) -> str:
    return f"{x*100:.2f}%"


def load_master(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="Master")
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales", "Customer ID"])
    # Ensure Year-Month exists
    if "Year-Month" not in df.columns:
        df["Year-Month"] = df["Order Date"].dt.to_period("M").astype(str)
    return df


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def forecast_monthly_linear(monthly: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    # monthly: columns ["Year-Month", "Sales"] sorted ascending
    y = monthly["Sales"].to_numpy(dtype=float)
    t = np.arange(len(y), dtype=float)

    # Fit y = a*t + b
    a, b = np.polyfit(t, y, 1)
    t_future = np.arange(len(y), len(y) + horizon, dtype=float)
    y_future = a * t_future + b

    # Build future Year-Month labels
    last_period = pd.Period(monthly["Year-Month"].iloc[-1], freq="M")
    future_periods = [str(last_period + i) for i in range(1, horizon + 1)]

    out = pd.DataFrame({
        "Year-Month": future_periods,
        "Forecast_Sales": y_future
    })
    out["Model"] = "LinearTrend"
    return out


def backtest_last_k(monthly: pd.DataFrame, k: int = 6) -> dict:
    # Simple backtest: train on all-but-last-k, predict last-k with linear trend
    monthly = monthly.copy()
    y = monthly["Sales"].to_numpy(dtype=float)
    if len(y) <= k + 2:
        return {"MAPE": float("nan"), "k": k}

    train = monthly.iloc[:-k]
    test = monthly.iloc[-k:]

    t_train = np.arange(len(train), dtype=float)
    y_train = train["Sales"].to_numpy(dtype=float)
    a, b = np.polyfit(t_train, y_train, 1)

    t_test = np.arange(len(train), len(train) + len(test), dtype=float)
    pred = a * t_test + b

    return {
        "k": k,
        "MAPE": mape(test["Sales"].to_numpy(dtype=float), pred),
        "Test_Actual_Sum": float(test["Sales"].sum()),
        "Test_Pred_Sum": float(pred.sum()),
    }


def plot_monthly_with_forecast(monthly: pd.DataFrame, fc: pd.DataFrame, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(monthly["Year-Month"], monthly["Sales"], label="Actual")
    plt.plot(fc["Year-Month"], fc["Forecast_Sales"], label="Forecast")
    plt.xticks(rotation=90)
    plt.title("Monthly Sales: Actual + Forecast")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    # Use Order ID for frequency if available; otherwise count rows
    has_order_id = "Order ID" in df.columns
    snapshot = df["Order Date"].max() + pd.Timedelta(days=1)

    if has_order_id:
        freq = df.groupby("Customer ID")["Order ID"].nunique()
    else:
        freq = df.groupby("Customer ID")["Sales"].size()

    monetary = df.groupby("Customer ID")["Sales"].sum()
    recency = (snapshot - df.groupby("Customer ID")["Order Date"].max()).dt.days

    rfm = pd.DataFrame({
        "Customer ID": recency.index,
        "RecencyDays": recency.values,
        "Frequency": freq.reindex(recency.index).fillna(0).values,
        "Monetary": monetary.reindex(recency.index).fillna(0).values,
    })

    # Quintile scores (5=best)
    rfm["R_Score"] = pd.qcut(rfm["RecencyDays"], 5, labels=[5,4,3,2,1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    def segment(row) -> str:
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        if r >= 4 and f >= 3:
            return "Loyal"
        if r >= 4 and f <= 2:
            return "New / Promising"
        if r <= 2 and f >= 4:
            return "At Risk (High Frequency)"
        if r <= 2 and m >= 4:
            return "At Risk (High Value)"
        if r <= 2 and f <= 2:
            return "Hibernating"
        return "Needs Attention"

    rfm["Segment"] = rfm.apply(segment, axis=1)
    return rfm.sort_values(["Segment", "RFM_Score"], ascending=[True, False])


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data_clean/Superstore_Cleaned.xlsx")
    p.add_argument("--horizon", type=int, default=12)
    args = p.parse_args()

    df = load_master(Path(args.input))

    # Monthly series
    monthly = (
        df.groupby("Year-Month", as_index=False)["Sales"].sum()
          .sort_values("Year-Month")
    )
    Path("outputs/day3_tables").mkdir(parents=True, exist_ok=True)
    monthly.to_csv("outputs/day3_tables/monthly_sales.csv", index=False)

    # Forecast + backtest
    fc = forecast_monthly_linear(monthly, horizon=args.horizon)
    fc.to_csv("outputs/day3_forecast.csv", index=False)

    bt = backtest_last_k(monthly, k=6)

    # RFM
    rfm = build_rfm(df)
    rfm.to_csv("outputs/day3_rfm_segments.csv", index=False)

    # Plots
    plot_monthly_with_forecast(monthly, fc, Path("outputs/day3_charts/monthly_sales_forecast.png"))

    seg_summary = rfm.groupby("Segment", as_index=False).agg(
        Customers=("Customer ID", "count"),
        AvgMonetary=("Monetary", "mean"),
        TotalMonetary=("Monetary", "sum")
    ).sort_values("TotalMonetary", ascending=False)
    seg_summary.to_csv("outputs/day3_tables/rfm_segment_summary.csv", index=False)

    # Write summary markdown (talk track starter)
    total_sales = float(df["Sales"].sum())
    top20 = rfm.sort_values("Monetary", ascending=False).head(max(1, int(0.2 * len(rfm))))
    top20_share = float(top20["Monetary"].sum() / total_sales)

    lines = []
    lines.append("# Day 3 Forecast + Customer Segmentation")
    lines.append("")
    lines.append("## Forecast (Monthly Sales)")
    lines.append(f"- Model: Linear trend baseline")
    lines.append(f"- Backtest (last {bt['k']} months) MAPE: {pct(bt['MAPE']) if not np.isnan(bt['MAPE']) else 'N/A'}")
    lines.append("")
    lines.append("## Customer Concentration")
    lines.append(f"- Total Sales (all time): {money(total_sales)}")
    lines.append(f"- Top 20% customers contribute ~{pct(top20_share)} of sales (revenue concentration).")
    lines.append("")
    lines.append("## RFM Segments (Sales-only)")
    lines.append("- Use Recency (days since last order), Frequency (unique orders), Monetary (total sales).")
    lines.append("- See: `outputs/day3_rfm_segments.csv` and `outputs/day3_tables/rfm_segment_summary.csv`.")
    lines.append("")
    lines.append("## Draft Strategy Ideas (Revenue-focused)")
    lines.append("- **Retention:** target 'At Risk (High Value)' with reactivation offers and proactive outreach.")
    lines.append("- **Upsell:** target 'Loyal' & 'Champions' with bundles in top sub-categories.")
    lines.append("- **Ops:** use 'Ship Days' to identify slow delivery patterns and prioritize service improvements for high-value segments.")
    lines.append("")
    lines.append("## Files Generated")
    lines.append("- Forecast: `outputs/day3_forecast.csv`")
    lines.append("- RFM: `outputs/day3_rfm_segments.csv`")
    lines.append("- Chart: `outputs/day3_charts/monthly_sales_forecast.png`")

    write_md(Path("outputs/day3_forecast_summary.md"), lines)
    print("[DONE] Day 3 outputs generated.")


if __name__ == "__main__":
    main()
