# Step 6 – Power BI dashboard build guide

Data: `powerbi/data/` (exported by `src/06_export_powerbi.py`) · Build time: about 2–3 hours

## 1. Why a star schema
Power BI filters flow **from dimension tables to fact tables**. With dates, products and customers in
their own tables, one slicer (say Country = Germany) filters sales, cancellations and adjustments at
the same time. One wide flat table would make that slow and error-prone.

| Table | Grain (one row per…) | Rows | Use |
|---|---|---|---|
| `fact_sales` | product line on an invoice | 522,534 | revenue, customers, missing IDs |
| `fact_cancellations` | cancelled line | 8,665 | cancellation £ |
| `fact_writeoffs` | stock write-off line | 1,336 | write-off £, reasons |
| `fact_adjustments` | fee / manual / postage line | 2,945 | manual credits, Amazon fees |
| `dim_date` | day | 396 | time slicers, peak-season flag |
| `dim_product` | product (StockCode) | 3,926 | cancel rate, review flag |
| `dim_customer` | customer | 4,371 | segments, model scores, recommended action |
| `ref_*` (6 tables) | pre-computed summaries | small | model comparison, policy simulations |

`dim_customer` includes an `UNKNOWN` row for sales with no customer ID, and 37 customers who only
appear in cancellations/adjustments (they bought before Dec 2010), so every relationship resolves.

## 2. Load and model (Power BI Desktop)
1. **Get data → Text/CSV** → load all 13 files from `powerbi/data/`.
2. **Power Query – check types** (Transform data):
   * `Date` columns → **Date**; `InvoiceDate` → Date/Time
   * `CustomerID`, `StockCode`, `Invoice` → **Text** (important: IDs like `85123A` and `12346` must not become numbers)
   * £ and probability columns → Decimal number; `Has…`, `Is…`, `ReviewFlag`, `ReasonRecorded` → True/False
3. **Model view – relationships** (all one-to-many, single direction, from dimension to fact):

```
dim_date[Date]            ──1:*──> fact_sales[Date], fact_cancellations[Date],
                                    fact_writeoffs[Date], fact_adjustments[Date]
dim_product[StockCode]    ──1:*──> fact_sales[StockCode], fact_cancellations[StockCode],
                                    fact_writeoffs[StockCode]
dim_customer[CustomerID]  ──1:*──> fact_sales[CustomerID], fact_cancellations[CustomerID],
                                    fact_adjustments[CustomerID]
```
4. Mark `dim_date` as a date table (Table tools → Mark as date table → `Date`).
5. Sort `dim_date[Month]` by `dim_date[MonthStart]` (Column tools → Sort by column), or months show alphabetically.

## 3. DAX measures
Create a blank table called `_Measures` to hold them (Enter data → name it → OK).

```DAX
Product Sales = SUM ( fact_sales[Revenue] )

Cancelled £ = SUM ( fact_cancellations[CancelValue] )
Cancel Rate = DIVIDE ( [Cancelled £], [Product Sales] )

Write-off £ = SUM ( fact_writeoffs[EstValue] )
Unexplained Write-off % =
    DIVIDE ( CALCULATE ( [Write-off £], fact_writeoffs[ReasonRecorded] = FALSE () ), [Write-off £] )

Manual Credits £ =
    - CALCULATE ( SUM ( fact_adjustments[Revenue] ),
                  fact_adjustments[Type] = "Manual adjustment", fact_adjustments[IsCredit] = TRUE () )
Manual Credits ≥ £500 £ =
    - CALCULATE ( SUM ( fact_adjustments[Revenue] ),
                  fact_adjustments[Type] = "Manual adjustment", fact_adjustments[Revenue] <= -500 )

Amazon Fees £ = - CALCULATE ( SUM ( fact_adjustments[Revenue] ), fact_adjustments[Type] = "Amazon fee" )
Finance Charges £ =
    - CALCULATE ( SUM ( fact_adjustments[Revenue] ),
                  fact_adjustments[Type] IN { "Bad debt", "Bank charges", "Charity commission", "Discount" } )

Total Leakage £ = [Cancelled £] + [Write-off £] + [Manual Credits £] + [Amazon Fees £] + [Finance Charges £]
Leakage % of Sales = DIVIDE ( [Total Leakage £], [Product Sales] )

Unattributed Sales £ = CALCULATE ( [Product Sales], fact_sales[HasCustomer] = FALSE () )
Unattributed % = DIVIDE ( [Unattributed Sales £], [Product Sales] )

Active Customers = CALCULATE ( DISTINCTCOUNT ( fact_sales[CustomerID] ), fact_sales[HasCustomer] = TRUE () )
Orders = DISTINCTCOUNT ( fact_sales[Invoice] )
Avg Order Value = DIVIDE ( [Product Sales], [Orders] )

Churn £ at Risk = SUM ( dim_customer[ChurnGBPAtRisk] )
Cancel £ Expected = SUM ( dim_customer[CancelExpLossGBP] )
Scored Customers = COUNTROWS ( FILTER ( dim_customer, NOT ISBLANK ( dim_customer[ChurnProb] ) ) )
Actual Churn Rate = AVERAGE ( dim_customer[ActualChurned] )
Actual Cancel Rate = AVERAGE ( dim_customer[ActualCancelled] )
```
Format: £ measures → Currency (£ English UK), 0 decimals; % measures → Percentage, 1 decimal.

**Check your build against these totals** (no filters): Product Sales £10,000,733 · Cancelled £229,415 ·
Write-off £364,557 · Manual Credits £146,784 · Amazon Fees £221,521 ·
Finance Charges £31,867 · **Total Leakage £994,143 (9.9% of sales)** · Unattributed % 15.1% · Scored Customers 2,783.

## 4. Pages
Keep one colour for data (e.g. blue), grey for baselines/comparisons, and red only for alerts.
Put slicers in one row across the top of each page.

### Page 1 – Executive overview: "Where is the money leaking?"
* **Slicers:** Month (dim_date[Month]), Country (dim_customer[Country])
* **KPI cards:** Product Sales · Total Leakage £ · Leakage % of Sales · Unattributed % · Active Customers · Churn £ at Risk
* **Bar chart:** `ref_leak_summary` – Leak vs Value, sorted descending
* **Line chart:** Product Sales by dim_date[Month]; filter `IsPartialMonth = False` on this visual
* **Text box:** the three headline recommendations (from the Step 7 memo)

### Page 2 – Process audit: "Which processes are failing?"
* **Bar:** Write-off £ by fact_writeoffs[Reason]; card: Unexplained Write-off %
* **Column:** Manual Credits £ by month; card: Manual Credits ≥ £500 £ (shows the 86% coverage of a £500 approval rule)
* **Column:** Unattributed % by month (the November spike)
* **Table:** dim_product filtered to `ReviewFlag = True` – Description, SalesGBP, CancelGBP, CancelRate (data bars on CancelGBP)

### Page 3 – Customer risk: "Who do we act on?"
* **Slicer:** dim_customer[Action]
* **Bar:** Churn £ at Risk by Action
* **Scatter:** x = ChurnProb, y = Typical90dSpend, size = ChurnGBPAtRisk, legend = Action
  (top-right = high risk, high value – the retention targets)
* **Table:** CustomerID, Country, Action, ChurnProb, CancelProb, Typical90dSpend, ChurnGBPAtRisk;
  conditional formatting on ChurnProb; sort by ChurnGBPAtRisk descending
* **Matrix:** `ref_campaign_sensitivity` – rows Save rate, columns Cost / contact, values Net £ (model)

### Page 4 – Model evidence: "Why trust the scores?"
* **Bar:** `ref_model_churn` – Model vs Test ROC-AUC (baseline highlighted grey)
* **Bar:** `ref_model_cancel` – Model vs Test PR-AUC
* **Table:** `ref_retention_targeting` filtered to Budget = top 10% – Strategy vs % of lost £ reached
* **Text box:** "Out-of-time test: scored 10 Sep 2011, outcomes 10 Sep – 9 Dec 2011. 7 models from
  6 families; Logistic Regression selected – tied for best accuracy, fastest and fully explainable."

## 5. Finishing touches
* Page names as questions (above) – reviewers read the dashboard as a story.
* Tooltips on the Page 3 scatter: CustomerID, Orders, SalesGBP, Action.
* Bookmark a "Top 10% retention list" view (Action slicer = 3. Retention contact).
* Export: File → Export → PDF for the resume portfolio; screenshot Page 1 for the README.

**No Power BI Desktop (e.g. on a Mac)?** Use Power BI in the browser (app.powerbi.com, free with a
university account), or build the same pages in Tableau Public with these CSVs – the design and
measures translate directly.
