from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ----------------------------
# Formatting helpers
# ----------------------------
def money(x: float) -> str:
    return f"${x:,.2f}"


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


# ----------------------------
# Data loading / monthly series
# ----------------------------
def load_master(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="Master")
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

    required = ["Order Date", "Sales", "Customer ID"]
    df = df.dropna(subset=required)

    if "Year-Month" not in df.columns:
        df["Year-Month"] = df["Order Date"].dt.to_period("M").astype(str)

    return df


def build_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    # Build a continuous monthly series (fills missing months with 0 sales).
    p = df["Order Date"].dt.to_period("M")
    s = df.groupby(p)["Sales"].sum().sort_index()
    s = s.asfreq("M", fill_value=0.0)  # continuous months

    out = s.reset_index()
    out.columns = ["Period", "Sales"]
    out["Year-Month"] = out["Period"].astype(str)
    return out[["Year-Month", "Sales"]]


# ----------------------------
# Forecast models
# ----------------------------
@dataclass
class ForecastModel:
    name: str
    predict: Callable[[pd.DataFrame, int], pd.DataFrame]


def forecast_linear_trend(train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    y = train["Sales"].to_numpy(dtype=float)
    t = np.arange(len(y), dtype=float)

    a, b = np.polyfit(t, y, 1)
    t_future = np.arange(len(y), len(y) + horizon, dtype=float)
    y_future = a * t_future + b
    y_future = np.clip(y_future, 0.0, None)

    last_period = pd.Period(train["Year-Month"].iloc[-1], freq="M")
    future_periods = [str(last_period + i) for i in range(1, horizon + 1)]

    return pd.DataFrame({"Year-Month": future_periods, "Forecast_Sales": y_future})


def forecast_seasonal_naive_12(train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    y = train["Sales"].to_numpy(dtype=float)
    season = 12

    if len(y) < season:
        return forecast_linear_trend(train, horizon)

    last_season = y[-season:]
    reps = int(np.ceil(horizon / season))
    y_future = np.tile(last_season, reps)[:horizon]
    y_future = np.clip(y_future, 0.0, None)

    last_period = pd.Period(train["Year-Month"].iloc[-1], freq="M")
    future_periods = [str(last_period + i) for i in range(1, horizon + 1)]

    return pd.DataFrame({"Year-Month": future_periods, "Forecast_Sales": y_future})


def forecast_trend_plus_month_seasonality(train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    OLS regression: Sales ~ intercept + time_index + month_of_year dummies
    This usually outperforms naive baselines on retail monthly data.
    """
    ym = pd.PeriodIndex(train["Year-Month"], freq="M")
    y = train["Sales"].to_numpy(dtype=float)
    t = np.arange(len(train), dtype=float)
    month = ym.month

    # X: [1, t, month2..month12] (month1 is baseline)
    X_parts = [np.ones(len(train)), t]
    for m in range(2, 13):
        X_parts.append((month == m).astype(float))
    X = np.vstack(X_parts).T

    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    last = ym[-1]
    future = [last + i for i in range(1, horizon + 1)]
    t_f = np.arange(len(train), len(train) + horizon, dtype=float)
    month_f = np.array([p.month for p in future])

    Xf_parts = [np.ones(horizon), t_f]
    for m in range(2, 13):
        Xf_parts.append((month_f == m).astype(float))
    Xf = np.vstack(Xf_parts).T

    y_future = Xf @ beta
    y_future = np.clip(y_future, 0.0, None)

    return pd.DataFrame({"Year-Month": [str(p) for p in future], "Forecast_Sales": y_future})


def forecast_exponential_smoothing(train: pd.DataFrame, horizon: int, 
                                    alpha: float = 0.3, beta: float = 0.1, gamma: float = 0.1) -> pd.DataFrame:
    """
    Holt-Winters exponential smoothing with trend and seasonality.
    More realistic for retail sales with growth bounds.
    """
    y = train["Sales"].to_numpy(dtype=float)
    n = len(y)
    season_length = 12
    
    if n < 2 * season_length:
        return forecast_trend_plus_month_seasonality(train, horizon)
    
    # Initialize seasonal components
    seasonals = {}
    for i in range(season_length):
        seasonals[i] = np.mean(y[i::season_length]) / np.mean(y)
    
    # Initialize level and trend
    level = y[0] / seasonals[0 % season_length]
    trend = np.mean(y[1:season_length]) - np.mean(y[:season_length-1])
    
    # Fit using Holt-Winters
    for i in range(n):
        seasonal_idx = i % season_length
        
        # Update level
        level_new = alpha * (y[i] / seasonals[seasonal_idx]) + (1 - alpha) * (level + trend)
        
        # Update trend
        trend = beta * (level_new - level) + (1 - beta) * trend
        
        # Update seasonality
        seasonals[seasonal_idx] = gamma * (y[i] / level_new) + (1 - gamma) * seasonals[seasonal_idx]
        
        level = level_new
    
    # Forecast with bounds
    y_future = []
    last_level = level
    last_trend = trend
    
    for i in range(horizon):
        seasonal_idx = (n + i) % season_length
        forecast_val = (last_level + (i + 1) * last_trend) * seasonals[seasonal_idx]
        
        # Apply realistic growth constraints (±30% from last actual)
        last_actual = y[-1]
        max_growth = 1.30
        min_growth = 0.70
        
        # Relax constraints further out
        horizon_factor = 1 + (i / horizon) * 0.5  # 50% more relaxed at end
        forecast_val = np.clip(forecast_val, last_actual * min_growth / horizon_factor, 
                               last_actual * max_growth * horizon_factor)
        
        y_future.append(forecast_val)
    
    y_future = np.array(y_future)
    y_future = np.clip(y_future, 0.0, None)
    
    last_period = pd.Period(train["Year-Month"].iloc[-1], freq="M")
    future_periods = [str(last_period + i) for i in range(1, horizon + 1)]
    
    return pd.DataFrame({"Year-Month": future_periods, "Forecast_Sales": y_future})


def smooth_forecast_transition(monthly: pd.DataFrame, forecast: pd.DataFrame, 
                                blend_months: int = 3) -> pd.DataFrame:
    """
    Smooth the transition from actual to forecast to avoid abrupt jumps.
    Blends the last actual value with the first forecast values.
    """
    y = monthly["Sales"].to_numpy(dtype=float)
    last_actual = y[-1]
    
    fc_values = forecast["Forecast_Sales"].to_numpy(dtype=float).copy()
    
    # Apply smooth blending for first blend_months
    for i in range(min(blend_months, len(fc_values))):
        # Weight decreases for actual, increases for forecast
        # Month 0: 70% actual, 30% forecast
        # Month 1: 40% actual, 60% forecast  
        # Month 2: 10% actual, 90% forecast
        weight_actual = max(0, 0.7 - i * 0.3)
        weight_forecast = 1 - weight_actual
        
        fc_values[i] = weight_actual * last_actual + weight_forecast * fc_values[i]
    
    # Ensure no negative values
    fc_values = np.clip(fc_values, 0.0, None)
    
    result = forecast.copy()
    result["Forecast_Sales"] = fc_values
    return result


def forecast_weighted_recent(train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Weighted average giving more importance to recent data.
    Better captures recent trends and seasonality.
    """
    y = train["Sales"].to_numpy(dtype=float)
    n = len(y)
    
    if n < 24:
        return forecast_trend_plus_month_seasonality(train, horizon)
    
    # Calculate weights (exponential decay)
    weights = np.exp(np.linspace(-1, 0, n))
    weights /= weights.sum()
    
    # Calculate trend from recent 12 months vs previous 12
    recent_12 = y[-12:]
    prev_12 = y[-24:-12]
    
    recent_mean = np.mean(recent_12)
    prev_mean = np.mean(prev_12)
    trend_factor = recent_mean / prev_mean if prev_mean > 0 else 1.0
    
    # Cap trend factor for realism
    trend_factor = np.clip(trend_factor, 0.85, 1.15)
    
    # Build forecast using seasonal pattern from recent year
    y_future = []
    for i in range(horizon):
        # Get same month from last year
        month_idx = (n - 12 + i) % 12
        base_value = recent_12[month_idx] if month_idx < 12 else recent_mean
        
        # Apply trend
        months_ahead = i + 1
        forecast_val = base_value * (trend_factor ** (months_ahead / 12))
        
        # Smooth transition for first few months
        if i < 3:
            blend_factor = (3 - i) / 3
            last_actual = y[-1]
            forecast_val = blend_factor * last_actual + (1 - blend_factor) * forecast_val
        
        y_future.append(forecast_val)
    
    y_future = np.array(y_future)
    y_future = np.clip(y_future, 0.0, None)
    
    last_period = pd.Period(train["Year-Month"].iloc[-1], freq="M")
    future_periods = [str(last_period + i) for i in range(1, horizon + 1)]
    
    return pd.DataFrame({"Year-Month": future_periods, "Forecast_Sales": y_future})


# ----------------------------
# Backtesting / model selection
# ----------------------------
def rolling_backtest(
    monthly: pd.DataFrame,
    model: ForecastModel,
    horizon: int = 3,
    min_train: int = 24,
    step: int = 1,
) -> Tuple[float, int]:
    y = monthly["Sales"].to_numpy(dtype=float)
    if len(y) < (min_train + horizon + 1):
        return float("nan"), 0

    scores = []
    for end in range(min_train, len(y) - horizon + 1, step):
        train = monthly.iloc[:end].copy()
        test = monthly.iloc[end : end + horizon].copy()

        pred = model.predict(train, horizon=horizon)["Forecast_Sales"].to_numpy(dtype=float)
        scores.append(mape(test["Sales"].to_numpy(dtype=float), pred))

    return float(np.mean(scores)), len(scores)


def select_best_model(monthly: pd.DataFrame, models: list[ForecastModel]) -> Tuple[ForecastModel, pd.DataFrame]:
    rows = []
    for m in models:
        rb_mape, n = rolling_backtest(monthly, m, horizon=3, min_train=24, step=1)
        rows.append({"Model": m.name, "RollingMAPE_h3": rb_mape, "Backtests": n})

    comp = pd.DataFrame(rows).sort_values("RollingMAPE_h3", ascending=True)
    best_name = comp.iloc[0]["Model"]
    best = next(x for x in models if x.name == best_name)
    return best, comp


# ----------------------------
# Plotting
# ----------------------------
def plot_monthly_with_forecast(monthly: pd.DataFrame, fc: pd.DataFrame, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    x_actual = pd.PeriodIndex(monthly["Year-Month"], freq="M").to_timestamp()
    x_fore = pd.PeriodIndex(fc["Year-Month"], freq="M").to_timestamp()

    plt.figure(figsize=(10, 5))
    plt.plot(x_actual, monthly["Sales"], label="Actual")
    plt.plot(x_fore, fc["Forecast_Sales"], label="Forecast", linestyle="--")

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right")
    plt.title("Monthly Sales: Actual + Forecast")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ----------------------------
# RFM
# ----------------------------
def _qscore(series: pd.Series, q: int = 5, higher_is_better: bool = True) -> pd.Series:
    # Robust quantile scoring (handles duplicates)
    ranked = series.rank(method="first")
    bins = pd.qcut(ranked, q=q, labels=False, duplicates="drop")
    if bins.isna().all():
        return pd.Series(np.ones(len(series), dtype=int), index=series.index)
    # bins are 0..k-1
    if higher_is_better:
        return (bins + 1).astype(int)
    # invert so lower values get higher score
    k = int(bins.max()) + 1
    return (k - bins).astype(int)


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    has_order_id = "Order ID" in df.columns
    snapshot = df["Order Date"].max() + pd.Timedelta(days=1)

    freq = (
        df.groupby("Customer ID")["Order ID"].nunique()
        if has_order_id
        else df.groupby("Customer ID")["Sales"].size()
    )
    monetary = df.groupby("Customer ID")["Sales"].sum()
    recency = (snapshot - df.groupby("Customer ID")["Order Date"].max()).dt.days

    rfm = pd.DataFrame(
        {
            "Customer ID": recency.index,
            "RecencyDays": recency.values,
            "Frequency": freq.reindex(recency.index).fillna(0).values,
            "Monetary": monetary.reindex(recency.index).fillna(0).values,
        }
    )

    rfm["R_Score"] = _qscore(rfm["RecencyDays"], q=5, higher_is_better=False)  # lower days = better
    rfm["F_Score"] = _qscore(rfm["Frequency"], q=5, higher_is_better=True)
    rfm["M_Score"] = _qscore(rfm["Monetary"], q=5, higher_is_better=True)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    def segment(row) -> str:
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        if r >= 4 and f >= 3:
            return "Loyal"
        if r >= 4 and f <= 2:
            return "New / Promising"
        if r <= 2 and (f >= 4 or m >= 4):
            return "At Risk (High Value/Freq)"
        if r <= 2 and f <= 2:
            return "Hibernating"
        return "Needs Attention"

    order = [
        "Champions",
        "Loyal",
        "New / Promising",
        "Needs Attention",
        "At Risk (High Value/Freq)",
        "Hibernating",
    ]
    rfm["Segment"] = rfm.apply(segment, axis=1)
    rfm["Segment"] = pd.Categorical(rfm["Segment"], categories=order, ordered=True)
    return rfm.sort_values(["Segment", "RFM_Score"], ascending=[True, False])


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Day 3: Forecast (model selection) + RFM segmentation")
    p.add_argument("--input", default="data_clean/Superstore_Cleaned.xlsx")
    p.add_argument("--horizon", type=int, default=12)
    args = p.parse_args()

    out_tables = Path("outputs/day3_tables")
    out_charts = Path("outputs/day3_charts")
    out_tables.mkdir(parents=True, exist_ok=True)
    out_charts.mkdir(parents=True, exist_ok=True)

    df = load_master(Path(args.input))
    monthly = build_monthly_sales(df)
    monthly.to_csv(out_tables / "monthly_sales.csv", index=False)

    models = [
        ForecastModel("LinearTrend", forecast_linear_trend),
        ForecastModel("SeasonalNaive12", forecast_seasonal_naive_12),
        ForecastModel("Trend+MonthSeasonality", forecast_trend_plus_month_seasonality),
        ForecastModel("ExponentialSmoothing", forecast_exponential_smoothing),
        ForecastModel("WeightedRecent", forecast_weighted_recent),
    ]
    best, comp = select_best_model(monthly, models)
    comp.to_csv(out_tables / "model_backtest_comparison.csv", index=False)

    fc = best.predict(monthly, horizon=args.horizon)
    
    # Apply smooth transition from last actual to forecast
    fc = smooth_forecast_transition(monthly, fc)
    
    fc["Model"] = best.name
    fc.to_csv("outputs/day3_forecast.csv", index=False)

    plot_monthly_with_forecast(monthly, fc, out_charts / "monthly_sales_forecast.png")

    rfm = build_rfm(df)
    rfm.to_csv("outputs/day3_rfm_segments.csv", index=False)

    seg_summary = rfm.groupby("Segment", as_index=False, observed=False).agg(
        Customers=("Customer ID", "count"),
        AvgMonetary=("Monetary", "mean"),
        TotalMonetary=("Monetary", "sum"),
    ).sort_values("TotalMonetary", ascending=False)
    seg_summary.to_csv(out_tables / "rfm_segment_summary.csv", index=False)

    total_sales = float(df["Sales"].sum())
    top20 = rfm.sort_values("Monetary", ascending=False).head(max(1, int(0.2 * len(rfm))))
    top20_share = float(top20["Monetary"].sum() / total_sales)

    lines = []
    lines.append("# Day 3 Forecast + Customer Segmentation")
    lines.append("")
    lines.append("## Forecast (Monthly Sales)")
    lines.append(f"- Selected model: **{best.name}** (lowest RollingMAPE_h3)")
    best_row = comp[comp["Model"] == best.name].iloc[0]
    lines.append(f"- Rolling backtest MAPE (3-month horizon): {pct(float(best_row['RollingMAPE_h3']))} over {int(best_row['Backtests'])} tests")
    lines.append("")
    lines.append("## Customer Concentration")
    lines.append(f"- Total Sales (all time): {money(total_sales)}")
    lines.append(f"- Top 20% customers contribute ~{pct(top20_share)} of sales (revenue concentration).")
    lines.append("")
    lines.append("## RFM Segments (Sales-only)")
    lines.append("- Recency = days since last order (lower is better)")
    lines.append("- Frequency = unique orders (if Order ID exists), else row count")
    lines.append("- Monetary = total sales")
    lines.append("")
    lines.append("## Files Generated")
    lines.append("- Monthly series: `outputs/day3_tables/monthly_sales.csv`")
    lines.append("- Model comparison: `outputs/day3_tables/model_backtest_comparison.csv`")
    lines.append("- Forecast: `outputs/day3_forecast.csv`")
    lines.append("- RFM: `outputs/day3_rfm_segments.csv`")
    lines.append("- Segment summary: `outputs/day3_tables/rfm_segment_summary.csv`")
    lines.append("- Chart: `outputs/day3_charts/monthly_sales_forecast.png`")

    write_md(Path("outputs/day3_forecast_summary.md"), lines)
    print("[DONE] Day 3 outputs generated.")


if __name__ == "__main__":
    main()
