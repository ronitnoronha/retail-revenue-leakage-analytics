# Step 1 – Data audit and cleaning

Script: `src/01_clean.py` · Outputs: `data/processed/`

## Why audit before cleaning
Every cleaning rule changes the numbers that end up on the resume. So first we looked at every
oddity in the raw file and decided, for each one, whether it is an error, a different kind of
transaction, or real business behaviour. Real behaviour is never deleted just because it looks
unusual.

## Rules, in order

| # | What we found | Method | Why | Rows affected |
|---|---|---|---|---|
| 1 | Invoice/StockCode contain letters ("C536379", "85123A"); file uses latin-1 £ signs | Read as text, `encoding="latin-1"` | Reading as numbers would crash or drop letters | – |
| 2 | Dates stored as text; Customer ID stored as float (17850.0); "Â£" in descriptions | `pd.to_datetime` with explicit format; cast ID to clean text; string replace | Explicit format avoids day/month mix-ups; IDs are labels, not numbers | all |
| 3 | 5,268 rows identical in every column | `drop_duplicates`, keep first | System double-logged the line; keeping both double-counts revenue | −5,268 |
| 4a | Non-product codes: POST, DOT, AMAZONFEE, M (Manual), BANK CHARGES, D (Discount), CRUK, vouchers, "Adjust bad debt" | Moved to `non_product` table | They are fees and accounting entries, not products. Kept separately for the process-audit step | 2,945 |
| 4b | Negative qty, price 0, no customer, notes like "damaged", "thrown away", "check" | Moved to `stock_writeoffs` | Warehouse stock adjustments, not customer returns | 1,336 |
| 4c | Invoices starting with "C" | Moved to `cancellations` | Customer cancellations/returns – the core of the leakage analysis | 8,668 |
| 5 | Sale lines with price 0 (98% have no customer) | Dropped | No revenue; samples or placeholders | −1,156 |
| 6 | Huge orders reversed minutes later with the identical quantity (80,995 paper birdies cancelled after 12 min; 74,215 jars after 16 min) | Matched purchase ↔ cancellation on customer, product, price, exact qty, ≤24h, qty ≥1,000; removed both sides | Keying errors. Left in, they fake ~£245k of "cancellations" | −3 sales, −3 cancels |
| 7 | 650 stock codes with several spellings | Use each code's most frequent description | Stops one product being split into several | – |
| 8 | Dec 2011 has only 9 days | `PartialMonth` flag | Prevents a fake "December crash" in trends | – |

## Deliberately NOT done
* **Missing Customer ID was not filled in.** You cannot guess who bought something. Those rows stay
  in revenue totals (it's real money) but are excluded from customer-level models. The missing rate
  itself becomes a finding: a data-capture process failure.
* **Large wholesale orders were not removed as outliers.** This retailer sells to wholesalers, so
  big orders are real. Only proven errors (rule 6) were removed.

## Result
| | Value |
|---|---|
| Product sales | £10.00M · 522,534 lines · 19,771 invoices |
| Product cancellations | £229k · 3,419 invoices · **2.3% of sales** |
| Sales with no customer ID | 25.1% of lines, 15.1% of revenue |
| Customers / countries | 4,333 / 38 |
| Biggest reversals outside products | Amazon fees −£235k, Manual adjustments −£147k |

**Correction to the earlier estimate:** the raw "£894k / 8.4% cancelled" figure was inflated by
keying errors, Amazon fee reversals and manual adjustments. Product cancellations are £229k (2.3%).
