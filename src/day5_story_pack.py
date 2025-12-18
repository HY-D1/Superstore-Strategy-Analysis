from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def money(x: float) -> str:
    return f"${x:,.0f}"


def pct(x: float) -> str:
    return f"{x*100:.1f}%"


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def first_existing_csv(*paths: str) -> pd.DataFrame | None:
    """
    Return the first successfully-read CSV from a list of candidate paths.
    (Avoids 'DataFrame truth value ambiguous' from using `or`.)
    """
    for p in paths:
        df = safe_read_csv(Path(p))
        if df is not None and not df.empty:
            return df
    return None


def top_row(df: pd.DataFrame | None, name_col: str, val_col: str = "Sales"):
    if df is None or df.empty or name_col not in df.columns or val_col not in df.columns:
        return ("N/A", float("nan"), float("nan"))
    d = df.sort_values(val_col, ascending=False).iloc[0]
    share = float(d["Share"]) if "Share" in df.columns else float("nan")
    return (str(d[name_col]), float(d[val_col]), share)


def main() -> None:
    p = argparse.ArgumentParser(description="Day 5: generate exec summary + slide outline from outputs/")
    p.add_argument("--out_exec", default="outputs/day5_executive_summary.md")
    p.add_argument("--out_slides", default="docs/day5_slide_outline.md")
    p.add_argument("--out_talk", default="docs/day5_talk_track.md")
    args = p.parse_args()

    # --- Inputs from Day 2/3 outputs ---
    monthly = first_existing_csv(
        "outputs/day2_tables/monthly_sales.csv",
        "outputs/day3_tables/monthly_sales.csv",
    )

    region = safe_read_csv(Path("outputs/day2_tables/sales_by_region.csv"))
    segment = safe_read_csv(Path("outputs/day2_tables/sales_by_segment.csv"))
    category = safe_read_csv(Path("outputs/day2_tables/sales_by_category.csv"))
    subcat = safe_read_csv(Path("outputs/day2_tables/sales_by_subcategory.csv"))
    ship = safe_read_csv(Path("outputs/day2_tables/shipping_mode_summary.csv"))

    rfm = safe_read_csv(Path("outputs/day3_rfm_segments.csv"))
    rfm_seg = safe_read_csv(Path("outputs/day3_tables/rfm_segment_summary.csv"))
    forecast = safe_read_csv(Path("outputs/day3_forecast.csv"))
    model_comp = safe_read_csv(Path("outputs/day3_tables/model_backtest_comparison.csv"))

    # --- Compute key metrics ---
    total_sales = float(monthly["Sales"].sum()) if monthly is not None and "Sales" in monthly.columns else float("nan")

    ym_min = monthly["Year-Month"].iloc[0] if monthly is not None and "Year-Month" in monthly.columns else "N/A"
    ym_max = monthly["Year-Month"].iloc[-1] if monthly is not None and "Year-Month" in monthly.columns else "N/A"

    top_region, top_region_sales, top_region_share = top_row(region, "Region")
    top_segment, top_segment_sales, top_segment_share = top_row(segment, "Segment")
    top_category, top_category_sales, top_category_share = top_row(category, "Category")

    top_subcat = "N/A"
    top_subcat_sales = float("nan")
    if subcat is not None and not subcat.empty and "Sub-Category" in subcat.columns and "Sales" in subcat.columns:
        r = subcat.sort_values("Sales", ascending=False).iloc[0]
        top_subcat = str(r["Sub-Category"])
        top_subcat_sales = float(r["Sales"])

    # Shipping best/worst
    ship_best = ship_worst = "N/A"
    ship_best_days = ship_worst_days = float("nan")
    if ship is not None and not ship.empty and "AvgShipDays" in ship.columns and "Ship Mode" in ship.columns:
        ship_sorted = ship.sort_values("AvgShipDays", ascending=True)
        ship_best = str(ship_sorted.iloc[0]["Ship Mode"])
        ship_best_days = float(ship_sorted.iloc[0]["AvgShipDays"])
        ship_worst = str(ship_sorted.iloc[-1]["Ship Mode"])
        ship_worst_days = float(ship_sorted.iloc[-1]["AvgShipDays"])

    # Customer concentration (Top 20% share) using RFM monetary (Sales-only)
    top20_share = float("nan")
    if rfm is not None and not rfm.empty and "Monetary" in rfm.columns:
        rfm2 = rfm.sort_values("Monetary", ascending=False).copy()
        n = len(rfm2)
        k = max(1, int(0.2 * n))
        denom = float(rfm2["Monetary"].sum())
        if denom != 0:
            top20_share = float(rfm2.head(k)["Monetary"].sum() / denom)

    # Forecast totals + growth vs last 12 months actual
    fc_total = float("nan")
    growth_vs_last12 = float("nan")
    if forecast is not None and not forecast.empty and "Forecast_Sales" in forecast.columns:
        fc_total = float(forecast["Forecast_Sales"].sum())

    if monthly is not None and "Sales" in monthly.columns and len(monthly) >= 12 and not np.isnan(fc_total):
        last12 = float(monthly["Sales"].tail(12).sum())
        if last12 != 0:
            growth_vs_last12 = (fc_total - last12) / last12

    # Model selection info (try to find a model + error column)
    best_model_line = "Model: baseline"
    if model_comp is not None and not model_comp.empty:
        # Pick first row as "best" if file is already sorted, else sort by first numeric error col
        error_cols = [c for c in model_comp.columns if "mape" in c.lower() or "error" in c.lower()]
        if error_cols:
            ec = error_cols[0]
            model_comp2 = model_comp.sort_values(ec, ascending=True)
            best = model_comp2.iloc[0]
            if "Model" in best.index:
                best_model_line = f"Model: {best['Model']} | {ec} = {pct(float(best[ec]))}"
        elif "Model" in model_comp.columns:
            best = model_comp.iloc[0]
            best_model_line = f"Model: {best['Model']} (selected by comparison table)"

    # --- Write Executive Summary ---
    out_exec = Path(args.out_exec)
    out_exec.parent.mkdir(parents=True, exist_ok=True)

    exec_md = []
    exec_md += ["# Executive Summary — Superstore Strategy Analysis", ""]
    exec_md += [f"**Coverage:** {ym_min} to {ym_max}", ""]
    exec_md += ["## Objective", "- Understand sales drivers, customer concentration, and shipping performance; create a forecast baseline and retention-focused segmentation.", ""]
    exec_md += ["## Key Findings (Sales-only dataset)",
                f"- **Total sales:** {money(total_sales)}" if not np.isnan(total_sales) else "- **Total sales:** N/A",
                f"- **Top region:** {top_region} ({pct(top_region_share) if not np.isnan(top_region_share) else 'N/A'} of sales)",
                f"- **Top segment:** {top_segment} ({pct(top_segment_share) if not np.isnan(top_segment_share) else 'N/A'} of sales)",
                f"- **Top category:** {top_category} ({pct(top_category_share) if not np.isnan(top_category_share) else 'N/A'} of sales)",
                f"- **Top sub-category:** {top_subcat} ({money(top_subcat_sales) if not np.isnan(top_subcat_sales) else 'N/A'})",
                f"- **Shipping speed:** best = {ship_best} ({ship_best_days:.1f} days), worst = {ship_worst} ({ship_worst_days:.1f} days)" if ship_best != "N/A" else "- **Shipping speed:** N/A",
                f"- **Customer concentration:** Top 20% customers contribute ~{pct(top20_share)} of sales" if not np.isnan(top20_share) else "- **Customer concentration:** N/A",
                ""]
    exec_md += ["## Forecast (Next 12 months)",
                f"- {best_model_line}",
                f"- Forecast total (next 12 months): {money(fc_total)}" if not np.isnan(fc_total) else "- Forecast total: N/A",
                f"- vs last 12 months actual: {pct(growth_vs_last12)}" if not np.isnan(growth_vs_last12) else "- vs last 12 months: N/A",
                ""]
    exec_md += ["## Recommendations (Actionable)",
                "- **Retention:** prioritize At-Risk high-value customers with reactivation offers and outreach.",
                "- **Growth:** upsell/bundle offers for Loyal + Champions in top-performing sub-categories.",
                "- **Ops:** reduce Ship Days for high-value segments (set shipping SLA targets by Ship Mode).",
                ""]
    exec_md += ["## Evidence (Repo outputs)",
                "- `outputs/day2_insights.md` (insights memo)",
                "- `outputs/day2_tables/` and `outputs/day2_charts/` (breakdowns + visuals)",
                "- `outputs/day3_forecast.csv` + `outputs/day3_charts/monthly_sales_forecast.png`",
                "- `outputs/day3_rfm_segments.csv` + `outputs/day3_tables/rfm_segment_summary.csv`",
                ""]
    out_exec.write_text("\n".join(exec_md), encoding="utf-8")

    # --- Slide Outline ---
    out_slides = Path(args.out_slides)
    out_slides.parent.mkdir(parents=True, exist_ok=True)

    slide_md = []
    slide_md += ["# Day 5 Slide Outline (7–8 slides)", ""]
    slide_md += ["## 1) Title", "- Superstore Strategy Analysis — Sales, Ops & Retention", ""]
    slide_md += ["## 2) Problem & Approach", "- Data cleaning + QA → insights → forecast baseline → RFM segmentation → dashboard", ""]
    slide_md += ["## 3) Sales Trend + Seasonality", "- Monthly sales trend chart + 1 key takeaway", ""]
    slide_md += ["## 4) Mix Drivers",
                 f"- Top Region: {top_region}",
                 f"- Top Category: {top_category}",
                 f"- Top Sub-Category: {top_subcat}",
                 ""]
    slide_md += ["## 5) Customer Concentration + RFM",
                 f"- Top 20% customers share: {pct(top20_share) if not np.isnan(top20_share) else 'N/A'}",
                 "- RFM segments: Champions / Loyal / At-Risk / Hibernating",
                 ""]
    slide_md += ["## 6) Forecast Baseline",
                 f"- {best_model_line}",
                 f"- Next 12 months forecast total: {money(fc_total) if not np.isnan(fc_total) else 'N/A'}",
                 ""]
    slide_md += ["## 7) Recommendations + Expected Impact",
                 "- Retention playbook by segment (At-Risk / Loyal / Champions)",
                 "- Shipping SLA improvement targets by Ship Mode",
                 ""]
    slide_md += ["## 8) Close",
                 "- What I built, tools used, and links (repo + Power BI)",
                 ""]
    out_slides.write_text("\n".join(slide_md), encoding="utf-8")

    # --- Talk Track ---
    out_talk = Path(args.out_talk)
    out_talk.parent.mkdir(parents=True, exist_ok=True)

    talk = []
    talk += ["# Day 5 Talk Track (5–8 minutes)", ""]
    talk += ["## Opening (20s)",
             "- I built an end-to-end strategy analysis pipeline on Superstore data to identify sales drivers, retention opportunities, and shipping/ops improvements.",
             ""]
    talk += ["## Data + QA (30s)",
             "- Day 1: cleaned the dataset, standardized dates, created Year-Month and Ship Days, and generated a QA report.",
             ""]
    talk += ["## Insights (1.5–2 min)",
             f"- Total sales: {money(total_sales)}" if not np.isnan(total_sales) else "- Total sales: N/A",
             f"- Biggest region driver: {top_region} ({pct(top_region_share) if not np.isnan(top_region_share) else 'N/A'})",
             f"- Biggest segment driver: {top_segment} ({pct(top_segment_share) if not np.isnan(top_segment_share) else 'N/A'})",
             f"- Shipping: best {ship_best} vs worst {ship_worst} (service gap).",
             ""]
    talk += ["## Customers (1–1.5 min)",
             f"- Top 20% customers contribute ~{pct(top20_share)} of sales → retention focus." if not np.isnan(top20_share) else "- Customer concentration: N/A",
             "- RFM segments identify Champions/Loyal/At-Risk cohorts for targeted actions.",
             ""]
    talk += ["## Forecast (1 min)",
             f"- Forecast baseline selected by backtest: {best_model_line}",
             f"- Next 12 months forecast total: {money(fc_total) if not np.isnan(fc_total) else 'N/A'}",
             ""]
    talk += ["## Recommendations (1 min)",
             "- Retention: reactivation outreach for At-Risk high-value customers.",
             "- Growth: bundles/upsells for Loyal + Champions in top sub-categories.",
             "- Ops: reduce Ship Days and prioritize high-value cohorts.",
             ""]
    talk += ["## Close (20s)",
             "- Deliverables: reproducible scripts + tables/charts + Power BI Service dashboard screenshots/link in README.",
             ""]
    out_talk.write_text("\n".join(talk), encoding="utf-8")

    print("[DONE] Day 5 generated: executive summary + slide outline + talk track.")


if __name__ == "__main__":
    main()
