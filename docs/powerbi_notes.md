# Power BI Dashboard Notes (Day 4)

## Data
- Source: `outputs/bi/superstore_bi.csv`
- Includes: cleaned master + RFM segments joined on Customer ID (Day 3 output)

## Model
- Date table created with CALENDAR based on Order Date
- Relationship: Date[Date] -> superstore_bi[Order Date]

## Key Measures (DAX)
- Total Sales = SUM(superstore_bi[Sales])
- Orders = DISTINCTCOUNT(superstore_bi[Order ID])
- Customers = DISTINCTCOUNT(superstore_bi[Customer ID])
- AOV = DIVIDE([Total Sales], [Orders])
- Avg Ship Days = AVERAGE(superstore_bi[Ship Days])

## Page Layout
- KPI cards: Total Sales, Orders, Customers, AOV, Avg Ship Days
- Trend: Total Sales by YearMonth
- Mix: Sales by Region, Category/Sub-Category
- Ops: Avg Ship Days by Ship Mode
- Customers: Sales by RFM Segment
- Slicers: Year, Region, Category (and Segment optional)
