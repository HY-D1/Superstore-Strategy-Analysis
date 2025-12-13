from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------
# Formatting helpers
# --------------------------
def money(x: float) -> str:
    return f"${x:,.2f}"


def pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x * 100:.2f}%"


# --------------------------
# Data loading & monthly series
# --------------------------
def load_master(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="Master")
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

    # Sales-only schema is OK; we need at least these
    df = df.dropna(subset=["Order Date", "Sales", "Customer ID"])

    # Ensure Year-Month and Ship Days exist
    if "Year-Month" not in df.columns:
        df["Year-Month"] = df["Order Date"].dt.to_period("M").astype(str)
    if "Ship Days" not in df.columns and "Ship Date" in df.columns:
        df["Ship Days"] = (df["Ship Date"] - df["Order Date"]).dt.days

    return df


def make_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a continuous monthly time series with PeriodIndex.
    Fills missing months with 0 sales (rare but makes modeling stable).
    """
    monthly_raw = (
        df.groupby("Year-Month", as_index=False)["Sales"]
        .sum()
    )

    periods = pd.PeriodIndex(monthly_raw["Year-Month"], freq="M")
    monthly = pd.DataFrame({"Period": periods, "Sales": monthly_raw["Sales"].astype(float).to_numpy()})
    monthly = monthly.sort_values("Period").set_index("Period")

    full_idx = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(full_idx)
    monthly["Sales"] = monthly["Sales"].fillna(0.0)

    monthly["Year-Month"] = monthly.index.astype(str)
    monthly = monthly.reset_index().rename(columns={"index": "Period"})  # keep Period column

    return monthly[["Period", "Year-Month", "Sales"]]


# --------------------------
# Metrics
# --------------------------
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


# --------------------------
# Forecast models
# --------------------------
def forecast_linear(monthly_train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    y = monthly_train["Sales"].to_numpy(dtype=float)
    t = np.arange(len(y), dtype=float)

    a, b = np.polyfit(t, y, 1)
    t_future = np.arange(len(y), len(y) + horizon, dtype=float)
    y_future = a * t_future + b
    y_future = np.clip(y_future, 0, None)

    last_period = monthly_train["Period"].iloc[-1]
    future_periods = [last_period + i for i in range(1, horizon + 1)]

    return pd.DataFrame({
        "Year-Month": [str(p) for p in future_periods],
        "Forecast_Sales": y_future,
        "Model": "Linear"
    })


def forecast_seasonal_naive(monthly_train: pd.DataFrame, horizon: int, season: int = 12) -> pd.DataFrame:
    """
    Repeat the last season (e.g., same month last year).
    """
    y = monthly_train["Sales"].to_numpy(dtype=float)

    if len(y) < season:
        return forecast_linear(monthly_train, horizon)

    last_season = y[-season:]
    reps = int(np.ceil(horizon / season))
    y_future = np.tile(last_season, reps)[:horizon]

    last_period = monthly_train["Period"].iloc[-1]
    future_periods = [last_period + i for i in range(1, horizon + 1)]

    return pd.DataFrame({
        "Year-Month": [str(p) for p in future_periods],
        "Forecast_Sales": y_future,
        "Model": f"SeasonalNaive({season})"
    })


def forecast_seasonal_avg(monthly_train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Average sales for each month-of-year across history (smoother than naive).
    """
    p = pd.PeriodIndex(monthly_train["Year-Month"], freq="M")
    tmp = monthly_train.copy()
    tmp["MonthOfYear"] = p.month

    month_avg = tmp.groupby("MonthOfYear")["Sales"].mean().to_dict()
    overall_mean = float(tmp["Sales"].mean())

    last_period = monthly_train["Period"].iloc[-1]
    future_periods = [last_period + i for i in range(1, horizon + 1)]
    y_future = [float(month_avg.get(p.month, overall_mean)) for p in future_periods]

    return pd.DataFrame({
        "Year-Month": [str(p) for p in future_periods],
        "Forecast_Sales": np.clip(np.array(y_future), 0, None),
        "Model": "SeasonalAvg"
    })


def forecast_seasonal_regression(monthly_train: pd.DataFrame, horizon: int, winsorize: bool = True) -> pd.DataFrame:
    """
    Strong simple model:
      Sales = intercept + trend*t + month-of-year effects (one-hot)  [additive]
    Often beats linear/naive for retail seasonality without heavy ML libs.
    """
    y = monthly_train["Sales"].to_numpy(dtype=float)
    periods = pd.PeriodIndex(monthly_train["Year-Month"], freq="M")
    months = periods.month
    t = np.arange(len(y), dtype=float)

    y_fit = y.copy()
    if winsorize and len(y_fit) >= 20:
        lo, hi = np.quantile(y_fit, [0.05, 0.95])
        y_fit = np.clip(y_fit, lo, hi)

    # Design matrix: [1, t, month_1..month_11] (month_12 is baseline to avoid collinearity)
    X_parts = [np.ones(len(y_fit)), t]
    for m in range(1, 12):
        X_parts.append((months == m).astype(float))
    X = np.column_stack(X_parts)

    beta, *_ = np.linalg.lstsq(X, y_fit, rcond=None)

    last_period = monthly_train["Period"].iloc[-1]
    future_periods = [last_period + i for i in range(1, horizon + 1)]
    t_future = np.arange(len(y_fit), len(y_fit) + horizon, dtype=float)
    months_future = np.array([p.month for p in future_periods], dtype=int)

    Xf_parts = [np.ones(horizon), t_future]
    for m in range(1, 12):
        Xf_parts.append((months_future == m).astype(float))
    Xf = np.column_stack(Xf_parts)

    y_future = Xf @ beta
    y_future = np.clip(y_future, 0, None)

    return pd.DataFrame({
        "Year-Month": [str(p) for p in future_periods],
        "Forecast_Sales": y_future,
        "Model": "SeasonalRegression"
    })


def make_forecast(monthly_train: pd.DataFrame, horizon: int, model: str) -> pd.DataFrame:
    model = model.lower()
    if model == "linear":
        return forecast_linear(monthly_train, horizon)
    if model == "seasonal_naive":
        return forecast_seasonal_naive(monthly_train, horizon, season=12)
    if model == "seasonal_avg":
        return forecast_seasonal_avg(monthly_train, horizon)
    if model == "seasonal_regression":
        return forecast_seasonal_regression(monthly_train, horizon, winsorize=True)
    raise ValueError("Unknown model. Choose: linear, seasonal_naive, seasonal_avg, seasonal_regression")


# --------------------------
# Backtesting (rolling)
# --------------------------
def rolling_backtest(
    monthly: pd.DataFrame,
    horizon: int = 6,
    min_train: int = 24,
    step: int = 1,
    model: str = "seasonal_regression",
) -> Dict[str, float]:
    """
    Rolling-origin backtest:
    - Train on first min_train months
    - Predict next 'horizon' months
    - Move forward by 'step' and repeat
    Returns mean MAPE across splits.
    """
    monthly = monthly.sort_values("Period").reset_index(drop=True)
    n = len(monthly)
    if n < (min_train + horizon + 1):
        return {"model": model, "splits": 0, "mape_mean": float("nan")}

    mapes: List[float] = []
    splits = 0

    for train_end in range(min_train, n - horizon + 1, step):
        train = monthly.iloc[:train_end].copy()
        test = monthly.iloc[train_end: train_end + horizon].copy()

        fc = make_forecast(train, horizon=horizon, model=model)
        pred = fc["Forecast_Sales"].to_numpy(dtype=float)
        actual = test["Sales"].to_numpy(dtype=float)

        m = mape(actual, pred)
        if not np.isnan(m):
            mapes.append(m)
            splits += 1

    return {
        "model": model,
        "splits": splits,
        "mape_mean": float(np.mean(mapes)) if mapes else float("nan"),
    }


# --------------------------
# Plotting
# --------------------------
def plot_monthly_with_forecast(monthly: pd.DataFrame, fc: pd.DataFrame, out_png: Path) -> None:
    import matplotlib.dates as mdates

    out_png.parent.mkdir(parents=True, exist_ok=True)

    x_actual = pd.PeriodIndex(monthly["Year-Month"], freq="M").to_timestamp()
    x_fore = pd.PeriodIndex(fc["Year-Month"], freq="M").to_timestamp()

    plt.figure(figsize=(12, 5))
    plt.plot(x_actual, monthly["Sales"], label="Actual")
    plt.plot(x_fore, fc["Forecast_Sales"], label=f"Forecast ({fc['Model'].iloc[0]})", linestyle="--")

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right")

    plt.title("Monthly Sales: Actual + Forecast")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# --------------------------
# RFM Segmentation
# --------------------------
def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
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

    # Safer qcut
    rfm["R_Score"] = pd.qcut(rfm["RecencyDays"], 5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
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


def write_md(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------
# Main
# --------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Day 3: Forecast + RFM segmentation with model benchmarking (Sales-only friendly).")
    p.add_argument("--input", default="data_clean/Superstore_Cleaned.xlsx")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--backtest_horizon", type=int, default=6)
    p.add_argument("--min_train", type=int, default=24)
    p.add_argument("--model", default="auto", choices=["auto", "linear", "seasonal_naive", "seasonal_avg", "seasonal_regression"])
    args = p.parse_args()

    df = load_master(Path(args.input))
    monthly = make_monthly_series(df)

    Path("outputs/day3_tables").mkdir(parents=True, exist_ok=True)
    Path("outputs/day3_charts").mkdir(parents=True, exist_ok=True)

    monthly.to_csv("outputs/day3_tables/monthly_sales.csv", index=False)

    # Benchmark models with rolling backtest
    candidates = ["linear", "seasonal_naive", "seasonal_avg", "seasonal_regression"]
    bt_rows = []
    for m in candidates:
        bt = rolling_backtest(
            monthly,
            horizon=args.backtest_horizon,
            min_train=args.min_train,
            step=1,
            model=m,
        )
        bt_rows.append(bt)

    bt_df = pd.DataFrame(bt_rows).sort_values("mape_mean")
    bt_df.to_csv("outputs/day3_tables/model_backtest_comparison.csv", index=False)

    # Choose model
    if args.model == "auto":
        valid = bt_df.dropna(subset=["mape_mean"])
        best_model = valid.iloc[0]["model"] if len(valid) else "seasonal_regression"
    else:
        best_model = args.model

    # Forecast with chosen model
    fc = make_forecast(monthly, horizon=args.horizon, model=str(best_model))
    fc.to_csv("outputs/day3_forecast.csv", index=False)

    # Plot
    plot_monthly_with_forecast(monthly, fc, Path("outputs/day3_charts/monthly_sales_forecast.png"))

    # RFM
    rfm = build_rfm(df)
    rfm.to_csv("outputs/day3_rfm_segments.csv", index=False)

    seg_summary = rfm.groupby("Segment", as_index=False).agg(
        Customers=("Customer ID", "count"),
        AvgMonetary=("Monetary", "mean"),
        TotalMonetary=("Monetary", "sum")
    ).sort_values("TotalMonetary", ascending=False)
    seg_summary.to_csv("outputs/day3_tables/rfm_segment_summary.csv", index=False)

    # Summary markdown
    total_sales = float(df["Sales"].sum())
    top20 = rfm.sort_values("Monetary", ascending=False).head(max(1, int(0.2 * len(rfm))))
    top20_share = float(top20["Monetary"].sum() / total_sales)

    best_row = bt_df[bt_df["model"] == best_model].head(1)
    best_mape = float(best_row["mape_mean"].iloc[0]) if len(best_row) else float("nan")
    splits = int(best_row["splits"].iloc[0]) if len(best_row) else 0

    forecast_total = float(fc["Forecast_Sales"].sum())
    forecast_avg = float(fc["Forecast_Sales"].mean())

    lines = []
    lines.append("# Day 3 Forecast + Customer Segmentation")
    lines.append("")
    lines.append("## Forecast (Monthly Sales)")
    lines.append(f"- Models compared (rolling backtest): {', '.join(candidates)}")
    lines.append(f"- Selected model: **{best_model}**")
    lines.append(f"- Rolling backtest: horizon={args.backtest_horizon} months, min_train={args.min_train} months, splits={splits}")
    lines.append(f"- Avg MAPE (selected): {pct(best_mape)}")
    lines.append(f"- Next {args.horizon} months forecast total: {money(forecast_total)} (avg/month: {money(forecast_avg)})")
    lines.append("- See: `outputs/day3_tables/model_backtest_comparison.csv`")
    lines.append("")
    lines.append("## Customer Concentration")
    lines.append(f"- Total Sales (all time): {money(total_sales)}")
    lines.append(f"- Top 20% customers contribute ~{pct(top20_share)} of sales (revenue concentration).")
    lines.append("")
    lines.append("## RFM Segments (Sales-only)")
    lines.append("- Recency = days since last order; Frequency = unique orders; Monetary = total sales.")
    lines.append("- See: `outputs/day3_rfm_segments.csv` and `outputs/day3_tables/rfm_segment_summary.csv`.")
    lines.append("")
    lines.append("## Draft Strategy Ideas (Revenue-focused)")
    lines.append("- **Retention:** target 'At Risk (High Value)' with reactivation outreach.")
    lines.append("- **Upsell:** target 'Loyal' & 'Champions' with bundles in top sub-categories.")
    lines.append("- **Ops:** prioritize faster shipping for high-value segments where Ship Days are high.")
    lines.append("")
    lines.append("## Files Generated")
    lines.append("- Forecast: `outputs/day3_forecast.csv`")
    lines.append("- Backtest: `outputs/day3_tables/model_backtest_comparison.csv`")
    lines.append("- RFM: `outputs/day3_rfm_segments.csv`")
    lines.append("- Chart: `outputs/day3_charts/monthly_sales_forecast.png`")

    write_md(Path("outputs/day3_forecast_summary.md"), lines)
    print("[DONE] Day 3 outputs generated.")
    print(f"[DONE] Best model: {best_model} | Backtest MAPE: {best_mape}")


if __name__ == "__main__":
    main()
