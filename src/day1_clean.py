from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Minimum columns needed to run Day 1 cleaning on ANY Superstore-like file
REQUIRED_MIN_COLS = ["Order Date", "Ship Date", "Sales"]

# Optional columns (some Superstore versions don't have these)
OPTIONAL_NUMERIC_COLS = ["Profit", "Discount", "Quantity"]  # only used if present

DATE_COLS = ["Order Date", "Ship Date"]


@dataclass
class DuplicateCheck:
    name: str
    keys: List[str]
    duplicate_rows: int


def _normalize_columns(cols: List[str]) -> List[str]:
    return [c.strip() for c in cols]


def _assert_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_MIN_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    for c in DATE_COLS:
        if c not in df.columns:
            continue

        s = df[c].astype(str).str.strip()

        # Try common Superstore formats first
        d1 = pd.to_datetime(s, errors="coerce", format="%m/%d/%Y")  # US
        if d1.isna().mean() > 0.2:
            d2 = pd.to_datetime(s, errors="coerce", format="%d/%m/%Y")  # day-first
            if d2.isna().mean() < d1.isna().mean():
                df[c] = d2
            else:
                df[c] = pd.to_datetime(df[c], errors="coerce") # fallback parser
        else:
            df[c] = d1

    return df


def _coerce_numeric_if_present(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    # Always available fields (based on Order Date / Ship Date)
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Year-Month"] = df["Order Date"].dt.to_period("M").astype(str)  # YYYY-MM
    df["Ship Days"] = (df["Ship Date"] - df["Order Date"]).dt.days

    # Optional: Profit-based fields
    if "Profit" in df.columns:
        df["Profit Margin %"] = np.where(
            df["Sales"].fillna(0) != 0,
            df["Profit"] / df["Sales"],
            0.0,
        )
        df["Profit Flag"] = np.where(df["Profit"] < 0, "Loss", "Profit")
    else:
        df["Profit Margin %"] = np.nan
        df["Profit Flag"] = "N/A (no Profit column)"

    return df


def _duplicate_checks(df: pd.DataFrame) -> List[DuplicateCheck]:
    candidate_key_sets = [
        ("OrderID_ProductID", ["Order ID", "Product ID"]),
        ("OrderID_ProductName", ["Order ID", "Product Name"]),
        ("OrderID_LineItem", ["Order ID", "Product ID", "Sales"]),
        ("RowID", ["Row ID"]),
    ]

    results: List[DuplicateCheck] = []
    for name, keys in candidate_key_sets:
        if all(k in df.columns for k in keys):
            dup = int(df.duplicated(subset=keys, keep=False).sum())
            results.append(DuplicateCheck(name=name, keys=keys, duplicate_rows=dup))
    return results


def _qa_summary(df: pd.DataFrame, dup_checks: List[DuplicateCheck]) -> pd.DataFrame:
    total_rows = len(df)

    def blanks(col: str):
        return int(df[col].isna().sum()) if col in df.columns else "N/A"

    sales_sum = float(df["Sales"].sum(skipna=True)) if "Sales" in df.columns else 0.0

    # Profit metrics only if Profit exists
    if "Profit" in df.columns:
        profit_sum = float(df["Profit"].sum(skipna=True))
        overall_margin = (profit_sum / sales_sum) if sales_sum else 0.0
        loss_orders_count = int((df["Profit"] < 0).sum())
        loss_orders_pct = float((df["Profit"] < 0).mean()) if total_rows else 0.0
    else:
        profit_sum = "N/A"
        overall_margin = "N/A"
        loss_orders_count = "N/A"
        loss_orders_pct = "N/A"

    qa_items: List[Tuple[str, object]] = [
        ("rows_total", total_rows),
        ("columns_total", len(df.columns)),
        ("blanks_order_date", blanks("Order Date")),
        ("blanks_ship_date", blanks("Ship Date")),
        ("blanks_sales", blanks("Sales")),
        ("sales_total", sales_sum),
        ("has_profit", "Profit" in df.columns),
        ("profit_total", profit_sum),
        ("overall_profit_margin", overall_margin),
        ("loss_orders_count", loss_orders_count),
        ("loss_orders_pct", loss_orders_pct),
        ("has_discount", "Discount" in df.columns),
        ("has_quantity", "Quantity" in df.columns),
    ]

    for d in dup_checks:
        qa_items.append((f"duplicate_rows_{d.name}", d.duplicate_rows))
        qa_items.append((f"duplicate_keys_{d.name}", ", ".join(d.keys)))

    return pd.DataFrame(qa_items, columns=["metric", "value"])


def _write_markdown_report(qa_df: pd.DataFrame, out_path: Path) -> None:
    lines = ["# Day 1 QA Report", "", "## Summary", ""]
    for _, row in qa_df.iterrows():
        lines.append(f"- **{row['metric']}**: {row['value']}")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_excel(master: pd.DataFrame, qa: pd.DataFrame, out_xlsx: Path) -> None:
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        master.to_excel(writer, sheet_name="Master", index=False)
        qa.to_excel(writer, sheet_name="QA_Summary", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 1: Clean Superstore CSV into Excel + QA summary (schema-flexible).")
    parser.add_argument("--input", required=True, help="Path to train.csv")
    parser.add_argument("--output", required=True, help="Path to output .xlsx")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_xlsx = Path(args.output)
    out_md = Path("outputs/day1_qa_report.md")

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    print(f"[INFO] Reading: {in_path}")
    df = pd.read_csv(in_path, encoding_errors="ignore")
    df.columns = _normalize_columns(list(df.columns))

    _assert_required_columns(df)

    print("[INFO] Coercing dates...")
    df = _coerce_dates(df)

    print("[INFO] Coercing numeric columns if present...")
    df = _coerce_numeric_if_present(df, ["Sales"] + OPTIONAL_NUMERIC_COLS)

    print("[INFO] Adding derived fields...")
    df = _add_derived_fields(df)

    print("[INFO] Running duplicate checks...")
    dup_checks = _duplicate_checks(df)

    print("[INFO] Creating QA summary...")
    qa_df = _qa_summary(df, dup_checks)

    print(f"[INFO] Writing Excel: {out_xlsx}")
    _write_excel(df, qa_df, out_xlsx)

    print(f"[INFO] Writing QA report: {out_md}")
    _write_markdown_report(qa_df, out_md)

    print("\n=== QA SUMMARY (key metrics) ===")
    for key in ["rows_total", "columns_total", "sales_total", "has_profit", "has_discount", "has_quantity"]:
        val = qa_df.loc[qa_df["metric"] == key, "value"].iloc[0]
        print(f"- {key}: {val}")

    print("\n[DONE] Day 1 cleaning complete.")


if __name__ == "__main__":
    main()
