# Classic data warehousing challenge

You are an expert datawarehouse architect. 
I have a report which shows sales achieved status daily. 
One retailer is associated with multiple distributors
One distributor has multiple sales person.
there is indirect relation of sales person to retailer.
if a Distributor has 5 sales person then in fact table need to put 5 line items for that day.

| id | state | retailer_code | dealer_code | TSO Name | invoice_date | moving_sum_final_amount | billedoutlets | baseline | total_potential | percentage_of_baseline | status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | Odisha | 100 | D301 | ABC | 2026-06-11 00:00:00.000000 | 100.0 | 1 | 10000 | 9900.0 | 1.0 | red |
| 1 | Odisha | 100 | D301 | PQR | 2026-06-11 00:00:00.000000 | 100.0 | 1 | 10000 | 9900.0 | 1.0 | red |
| 2 | Odisha | 100 | D301 | XYZ | 2026-06-11 00:00:00.000000 | 100.0 | 1 | 10000 | 9900.0 | 1.0 | red |
| 3 | Odisha | 100 | D301 | GHI | 2026-06-11 00:00:00.000000 | 100.0 | 1 | 10000 | 9900.0 | 1.0 | red |
| 4 | Odisha | 100 | D301 | DEF | 2026-06-11 00:00:00.000000 | 100.0 | 1 | 10000 | 9900.0 | 1.0 | red |


Now I want to solve the number of rows generated for date grid with retailer, distributor, and salesperson 
so if I keep last 90 days snapshot in the system, then I end up having
90 days * ( 121000 retailer * avg 2 distributor) * average 5 salesperson = 108,900,000 (Ten Crore Eighty Nine Lakh), this number is too high to deal with to show in bold bi dashboard.

please help me with better approach to solve this number of line items in facts table

The problem you are running into is a classic data warehousing challenge: **granularity mismatch combined with a many-to-many relationship**.

By forcing the salesperson dimension directly into a daily snapshot fact table alongside retailers and distributors, you are **cross-joining the relationships**. If a retailer has 2 distributors, and each distributor has 5 salespeople, you are multiplying the same base sales metrics (like `moving_sum_final_amount` or `baseline`) 10 times for a single retailer on a single day. This exponentially inflates your row count and forces you to deal with **double-counting issues** when aggregating values up to the distributor or retailer level.

Here are the two best architectural approaches to solve this issue and bring that 108 million row count down drastically.

---

## Approach 1: The Bridge Table / Relationship Fact (Highly Recommended)

Instead of forcing the salesperson into the Daily Sales Fact table, you separate the **sales transactions/snapshots** from the **coverage territory assignments**.

Since the relationship between a salesperson and a retailer is *indirect* (mediated through the distributor), you should keep the Fact table at the grain of **Date + Retailer + Distributor**.

### 1. Daily Sales Snapshot Fact (Grain: Date + Retailer + Distributor)

Remove the salesperson from this table completely.

* **Row Count Impact:** 90 days × (121,000 retailers × 2 distributors) = **21,780,000 rows**.
* *Result:* You instantly **cut your row count by 80%** (a 5x reduction).

### 2. Distributor-Salesperson Assignment Dim/Bridge

Create a separate dimension or bridge table that maps which salespeople belong to which distributor. If salespeople change frequently, this can be a Slowly Changing Dimension (SCD Type 2) or a simple daily snapshot bridge.

* **Grain:** Date + Distributor + Salesperson
* **Row Count:** 90 days × (242,000 distributor connections × 5 salespeople) = ~1.2 million rows total.

### How it works in Bold BI / Power BI:

You relate the Fact table to the Bridge table via `dealer_code` (and date, if assignments change daily). When a user filters by a specific salesperson, the BI tool filters the bridge, which propagates down to filter the distributor's sales in the main fact table. If you need to attribute sales evenly to salespeople, you can handle the allocation logic dynamically in the BI layer rather than duplicating physical storage.

---

## Approach 2: Granularity Split (Aggregated Snapshot + Transactional Fact)

If your BI dashboard primarily looks at overall performance, status tracking ("black"), and 90-day trends, you don't need a single mammoth table doing everything. You split the architecture based on how the data is consumed.

```
       [Raw Transaction / Pipeline Data]
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
[Fact Daily Status Snapshot]  [Fact Salesperson Performance]
 Grain: Date+Retailer+Dealer   Grain: Date+Salesperson
 Row Count: ~21.7M rows        Row Count: ~100K rows
 (Used for Grid/Status Maps)   (Used for Sales Rep Leaderboards)

```

### 1. Fact_Daily_Retailer_Status (For the main Grid View)

* **Grain:** `invoice_date` + `retailer_code` + `dealer_code`
* **Metrics:** `moving_sum_final_amount`, `billedoutlets`, `baseline`, `status`
* **Rows:** **21.7 Million Rows**
* This table populates your heavy grid view dashboards. Since salesperson is removed, the statuses don't duplicate.

### 2. Fact_Salesperson_Daily_Performance (For Salesperson Deep Dives)

* **Grain:** `invoice_date` + `salesperson_id` + `dealer_code`
* **Metrics:** Total sales achieved by that rep for the day, target baselines.
* **Rows:** 90 days × (121,000 retailers × 2 / 5 reps per dealer) = A tiny fraction of the size.

---

## Architecture Comparison

| Feature | Current Approach | Approach 1 (Bridge Table) | Approach 2 (Split Facts) |
| --- | --- | --- | --- |
| **Physical Rows** | ~109 Million | **~21.8 Million** | **~22 Million (Combined)** |
| **Risk of Double Counting** | High (Requires `DISTINCT` or division) | Low (Handled via relationships) | Zero (Aggregated cleanly at source) |
| **Dashboard Performance** | Sluggish / High Memory | Fast (Star Schema optimized) | Fastest (Aggregated tables run instantly) |
| **Implementation Effort** | None (Current state) | Medium (Requires BI relationship setup) | Medium (Requires 2 ETL pipelines) |

## Summary Checklist for your ETL Next Steps:

1. **Drop the Salesperson column** from the main query filling this specific daily grid snapshot.
2. Group your source data by `invoice_date`, `retailer_code`, and `dealer_code`.
3. Let the `status` calculation happen at the *Distributor-Retailer* connection level, not the salesperson level.
4. If a user *must* click a salesperson and see their associated retailers, use Bold BI's cross-filtering features linked through a separate entity mapping table rather than flattening it all into one massive table.
