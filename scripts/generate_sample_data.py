# scripts/generate_sample_data.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def main(out_path: str = "data_raw/sample_train.csv", n: int = 120, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)

    order_dates = pd.date_range("2017-01-01", periods=60, freq="7D")
    ship_modes = ["Standard Class", "Second Class", "First Class", "Same Day"]
    segments = ["Consumer", "Corporate", "Home Office"]
    regions = ["West", "East", "Central", "South"]
    categories = ["Furniture", "Office Supplies", "Technology"]
    subcats = {
        "Furniture": ["Chairs", "Tables", "Bookcases"],
        "Office Supplies": ["Binders", "Paper", "Storage"],
        "Technology": ["Phones", "Accessories", "Copiers"],
    }
    states = ["California", "New York", "Texas", "Washington", "Florida"]
    cities = ["Los Angeles", "New York City", "Houston", "Seattle", "Miami"]

    rows = []
    for i in range(1, n + 1):
        od = pd.Timestamp(rng.choice(order_dates))

        ship_days = int(rng.integers(1, 8))
        sd = od + pd.Timedelta(days=ship_days)

        cat = rng.choice(categories)
        sc = rng.choice(subcats[cat])

        rows.append(
            {
                "Row ID": i,
                "Order ID": f"CA-{int(od.year)}-{100000 + i}",
                "Order Date": od.strftime("%m/%d/%Y"),
                "Ship Date": sd.strftime("%m/%d/%Y"),
                "Ship Mode": rng.choice(ship_modes),
                "Customer ID": f"C-{int(rng.integers(1000, 1100))}",
                "Customer Name": f"Customer {int(rng.integers(1, 60))}",
                "Segment": rng.choice(segments),
                "Country": "United States",
                "City": rng.choice(cities),
                "State": rng.choice(states),
                "Postal Code": int(rng.integers(10000, 99999)),
                "Region": rng.choice(regions),
                "Product ID": f"P-{int(rng.integers(10000, 20000))}",
                "Category": cat,
                "Sub-Category": sc,
                "Product Name": f"{sc} Item {int(rng.integers(1, 200))}",
                "Sales": float(np.round(rng.uniform(10, 1200), 2)),
            }
        )

    df = pd.DataFrame(rows)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[DONE] wrote {out} rows={len(df)}")


if __name__ == "__main__":
    main()
