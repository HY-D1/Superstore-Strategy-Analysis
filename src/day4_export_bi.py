from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Day 4: export a flattened CSV for Power BI (Master + RFM segments).")
    p.add_argument("--master_xlsx", default="data_clean/Superstore_Cleaned.xlsx")
    p.add_argument("--rfm_csv", default="outputs/day3_rfm_segments.csv")
    p.add_argument("--out_csv", default="outputs/bi/superstore_bi.csv")
    args = p.parse_args()

    master = pd.read_excel(args.master_xlsx, sheet_name="Master")
    master["Order Date"] = pd.to_datetime(master["Order Date"], errors="coerce")
    master["Ship Date"] = pd.to_datetime(master["Ship Date"], errors="coerce")

    # Load RFM if exists
    rfm_path = Path(args.rfm_csv)
    if rfm_path.exists():
        rfm = pd.read_csv(rfm_path)
        # keep only useful columns
        keep = ["Customer ID", "RecencyDays", "Frequency", "Monetary", "RFM_Score", "Segment"]
        rfm = rfm[[c for c in keep if c in rfm.columns]]
        out = master.merge(rfm, on="Customer ID", how="left")
    else:
        out = master.copy()
        out["Segment"] = "N/A"
        out["RFM_Score"] = None

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[DONE] Wrote {out_path} | rows={len(out):,} cols={len(out.columns)}")


if __name__ == "__main__":
    main()
