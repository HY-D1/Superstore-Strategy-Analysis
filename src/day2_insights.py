from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def money(x: float) -> str:
    return f"${x:,.2f}"


def pct(x: float) -> str:
    return f"{x*100:.1f}%"


def read_master(input_path: Path) -> pd.DataFrame:
    if input_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(input_path, sheet_name="Master")
    else:
        df = pd.read_csv(input_path, encoding_errors="ignore")

    # Ensure expected columns exist (based on your Day 1 output)
    required = ["Order Date", "Ship Date", "Sales", "Region", "Segment", "Category", "Sub-Category", "Ship Mode", "Customer ID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")

    # Coerce dates (should already be correct from Day 1, but keep safe)
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

    # Derived fields if not present
    if "Year" not in df.columns:
        df["Year"] = df["Order Date"].dt.year
    if "Year-Month" not in df.columns:
        df["Year-Month"] = df["Order Date"].dt.to_period("M").astype(str)
    if "Ship Days" not in df.columns:
        df["Ship Days"] = (df["Ship Date"] - df["Order Date"]).dt.days

    # Basic cleaning
    df = df.dropna(subset=["Order Date", "Ship Date", "Sales"])
    return df


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_line_chart(year_month, y, title: str, path: Path) -> None:
    import matplotlib.dates as mdates

    path.parent.mkdir(parents=True, exist_ok=True)

    x = pd.PeriodIndex(year_month, freq="M").to_timestamp()

    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_bar_chart(labels, values, title: str, path: Path, top_n: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if top_n is not None:
        labels = labels[:top_n]
        values = values[:top_n]

    plt.figure()
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 2: Generate insights, tables, and charts from Superstore Master data (Sales-only friendly).")
    parser.add_argument("--input", required=True, help="Path to data_clean/Superstore_Cleaned.xlsx (recommended) or CSV")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_md = Path("outputs/day2_insights.md")

    df = read_master(in_path)

    total_sales = float(df["Sales"].sum())
    years = sorted([int(y) for y in df["Year"].dropna().unique()])
    year_min, year_max = min(years), max(years)

    # Monthly sales trend
    monthly = (
        df.groupby("Year-Month", as_index=False)["Sales"]
          .sum()
          .sort_values("Year-Month")
    )
    save_table(monthly, Path("outputs/day2_tables/monthly_sales.csv"))
    save_line_chart(monthly["Year-Month"], monthly["Sales"], "Monthly Sales Trend", Path("outputs/day2_charts/monthly_sales_trend.png"))

    # Yearly sales + YoY growth
    yearly = df.groupby("Year", as_index=False)["Sales"].sum().sort_values("Year")
    yearly["YoY Growth"] = yearly["Sales"].pct_change()
    save_table(yearly, Path("outputs/day2_tables/yearly_sales.csv"))
    save_bar_chart(yearly["Year"].astype(str).tolist(), yearly["Sales"].tolist(), "Sales by Year", Path("outputs/day2_charts/sales_by_year.png"))

    # Region / Segment / Category mix
    region = df.groupby("Region", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    region["Share"] = region["Sales"] / total_sales
    save_table(region, Path("outputs/day2_tables/sales_by_region.csv"))
    save_bar_chart(region["Region"].tolist(), region["Sales"].tolist(), "Sales by Region", Path("outputs/day2_charts/sales_by_region.png"))

    segment = df.groupby("Segment", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    segment["Share"] = segment["Sales"] / total_sales
    save_table(segment, Path("outputs/day2_tables/sales_by_segment.csv"))
    save_bar_chart(segment["Segment"].tolist(), segment["Sales"].tolist(), "Sales by Segment", Path("outputs/day2_charts/sales_by_segment.png"))

    category = df.groupby("Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    category["Share"] = category["Sales"] / total_sales
    save_table(category, Path("outputs/day2_tables/sales_by_category.csv"))
    save_bar_chart(category["Category"].tolist(), category["Sales"].tolist(), "Sales by Category", Path("outputs/day2_charts/sales_by_category.png"))

    # Sub-category top/bottom
    subcat = df.groupby("Sub-Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    subcat["Share"] = subcat["Sales"] / total_sales
    save_table(subcat, Path("outputs/day2_tables/sales_by_subcategory.csv"))
    save_bar_chart(subcat["Sub-Category"].tolist(), subcat["Sales"].tolist(), "Top 10 Sub-Categories by Sales", Path("outputs/day2_charts/top10_subcategory_sales.png"), top_n=10)

    bottom10 = subcat.sort_values("Sales", ascending=True).head(10)
    save_table(bottom10, Path("outputs/day2_tables/bottom10_subcategory_sales.csv"))

    # Shipping: Ship mode & speed
    ship_mode = df.groupby("Ship Mode", as_index=False).agg(
        Sales=("Sales", "sum"),
        AvgShipDays=("Ship Days", "mean"),
        Orders=("Order ID", "nunique") if "Order ID" in df.columns else ("Sales", "size"),
    ).sort_values("Sales", ascending=False)
    ship_mode["SalesShare"] = ship_mode["Sales"] / total_sales
    save_table(ship_mode, Path("outputs/day2_tables/shipping_mode_summary.csv"))
    save_bar_chart(ship_mode["Ship Mode"].tolist(), ship_mode["AvgShipDays"].tolist(), "Average Ship Days by Ship Mode", Path("outputs/day2_charts/avg_ship_days_by_mode.png"))

    # Customer concentration
    cust = df.groupby(["Customer ID"], as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    cust["Share"] = cust["Sales"] / total_sales
    cust["CumShare"] = cust["Share"].cumsum()
    top10 = cust.head(10).copy()
    top50 = cust.head(50).copy()
    save_table(top10, Path("outputs/day2_tables/top10_customers_by_sales.csv"))
    save_table(top50, Path("outputs/day2_tables/top50_customers_by_sales.csv"))

    top10_share = float(top10["Sales"].sum() / total_sales)
    top50_share = float(top50["Sales"].sum() / total_sales)

    # Repeat purchase / churn proxy (latest year vs previous years)
    # Customers with orders in latest year
    df["Year"] = df["Order Date"].dt.year
    latest_year = int(df["Year"].max())
    prev_years = df[df["Year"] < latest_year]
    latest = df[df["Year"] == latest_year]

    customers_latest = set(latest["Customer ID"].unique())
    customers_prev = set(prev_years["Customer ID"].unique())

    new_customers = customers_latest - customers_prev
    returning_customers = customers_latest & customers_prev
    churned_customers = customers_prev - customers_latest

    # Write insights markdown (talk track)
    lines = []
    lines.append("# Day 2 Insights (Sales-Only Superstore)")
    lines.append("")
    lines.append("## Dataset Snapshot")
    lines.append(f"- Rows: {len(df):,}")
    lines.append(f"- Years covered: {year_min}–{year_max} (latest = {latest_year})")
    lines.append(f"- Total Sales: {money(total_sales)}")
    lines.append("")
    lines.append("## Core Findings (draft bullets)")
    lines.append("")
    lines.append("### 1) Growth & Seasonality")
    if len(yearly) >= 2:
        last_yoy = yearly["YoY Growth"].iloc[-1]
        lines.append(f"- Sales trend shows clear seasonality; latest YoY growth = {pct(last_yoy) if pd.notna(last_yoy) else 'N/A'}.")
    else:
        lines.append("- Sales trend shows seasonality (monthly peaks/troughs visible).")
    lines.append("")

    lines.append("### 2) Region Mix")
    lines.append(f"- Top region by sales: **{region.iloc[0]['Region']}** at {money(float(region.iloc[0]['Sales']))} ({pct(float(region.iloc[0]['Share']))} of total).")
    lines.append("")

    lines.append("### 3) Segment Mix")
    lines.append(f"- Largest segment: **{segment.iloc[0]['Segment']}** at {money(float(segment.iloc[0]['Sales']))} ({pct(float(segment.iloc[0]['Share']))}).")
    lines.append("")

    lines.append("### 4) Category / Sub-Category Concentration")
    lines.append(f"- Top category: **{category.iloc[0]['Category']}** at {money(float(category.iloc[0]['Sales']))} ({pct(float(category.iloc[0]['Share']))}).")
    lines.append(f"- Top sub-category: **{subcat.iloc[0]['Sub-Category']}** at {money(float(subcat.iloc[0]['Sales']))} ({pct(float(subcat.iloc[0]['Share']))}).")
    lines.append("")

    lines.append("### 5) Shipping Mode vs Speed (Ops Proxy)")
    fastest = ship_mode.sort_values("AvgShipDays").iloc[0]
    slowest = ship_mode.sort_values("AvgShipDays", ascending=False).iloc[0]
    lines.append(f"- Fastest ship mode: **{fastest['Ship Mode']}** avg {float(fastest['AvgShipDays']):.2f} days.")
    lines.append(f"- Slowest ship mode: **{slowest['Ship Mode']}** avg {float(slowest['AvgShipDays']):.2f} days.")
    lines.append("")

    lines.append("### 6) Customer Concentration")
    lines.append(f"- Top 10 customers contribute **{pct(top10_share)}** of total sales.")
    lines.append(f"- Top 50 customers contribute **{pct(top50_share)}** of total sales.")
    lines.append("")

    lines.append("### 7) Retention Proxy (Latest Year)")
    lines.append(f"- Latest year customers: {len(customers_latest):,}")
    lines.append(f"- Returning customers: {len(returning_customers):,}")
    lines.append(f"- New customers: {len(new_customers):,}")
    lines.append(f"- Churned (had prior orders but none in latest year): {len(churned_customers):,}")
    lines.append("")
    lines.append("## Files Generated")
    lines.append("- Charts: `outputs/day2_charts/`")
    lines.append("- Tables: `outputs/day2_tables/`")
    lines.append("- This report: `outputs/day2_insights.md`")
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] Wrote {out_md}")
    print("[DONE] Charts/tables saved under outputs/day2_charts and outputs/day2_tables")


if __name__ == "__main__":
    main()
