# Step 3 – Feature engineering

Script: `src/03_features.py` · Output: `data/processed/features.parquet`

## The problem this step solves
Models need **one row per thing we predict about**, with columns describing it. Our data is one row
per product line on an invoice. So we turn 391k customer sales lines into one row per customer, per
snapshot date.

## Snapshots: how we stop data leakage
A prediction is always made on a specific day. On that day you only know the past. So for every
**cutoff date**:

```
           features: everything BEFORE the cutoff          label: next 90 days
  |----------------------------------------------------|=======================|
2010-12-01                                          cutoff                cutoff + 90d
```

| Split | Cutoffs | Label windows | Rows |
|---|---|---|---|
| Train | 1 Mar, 1 Apr, 1 May, 1 Jun 2011 | end by 30 Aug 2011 | 8,898 (2,715 customers) |
| Test | 10 Sep 2011 | 10 Sep – 9 Dec 2011 (the last 90 days) | 2,783 |

* The test label window starts **after every training label window has ended**. The models are
  scored on a period they never saw – exactly how they'd be used in real life.
* Using 4 training snapshots instead of 1 gives 4× more training rows and teaches the model what
  customers look like at different points in the year.
* The same customer appears in several snapshots. In Step 4, cross-validation will be **grouped by
  customer**, so one customer is never in both the training and validation folds.
* The script contains `assert` checks that fail if any feature uses data from on or after the cutoff.

### Who is in each snapshot
Customers who bought at least once in the **180 days before the cutoff**. Customers silent for longer
have already left; including them would make churn trivially easy to "predict" and inflate scores.

## Labels (what we predict)
| Label | Definition | Train rate | Test rate |
|---|---|---|---|
| `churn_90d` | No purchase in the 90 days after the cutoff | 43–50% | 38% |
| `cancel_90d` | At least one cancellation in the 90 days after the cutoff | 16–19% | 19% |

Also stored (not used as inputs): `future_revenue_90d` and `future_cancel_value_90d`, to measure
£ impact in Step 5.

**Why 90 days:** in Step 2, 90% of gaps between repeat orders were under 111 days, so 90 days of
silence is a meaningful signal, not normal behaviour.

**Note on the test period:** it covers Sep–Dec, the peak season, so churn is lower (38%) than in
training. That's a realistic shift; a model that still works there is robust.

## The 29 features
| Group | Features | Why |
|---|---|---|
| RFM | recency_days, frequency, monetary, avg_order_value, tenure_days | The standard customer-value measures |
| Recent activity | orders/revenue in the last 30 and 90 days, revenue in the prior 90 days, revenue_trend | Is the customer speeding up or slowing down? |
| Purchase rhythm | mean_gap_days, std_gap_days, overdue_ratio (recency ÷ usual gap) | 60 days of silence is alarming for a weekly buyer, normal for a quarterly one |
| Basket shape | avg lines and units per order, n_products, avg_unit_price, wholesale_line_share (lines of 24+ units) | Separates retail shoppers from wholesale buyers |
| Cancellation history | n_cancel_invoices, cancel_value, cancel_rate, days_since_cancel | Past cancellers are likely to cancel again |
| Product risk | risky_product_share: share of spend on products with ≥5% past cancel rate | Some products get returned a lot |
| Process | n_manual_credits | Customers who needed manual fixes |
| Country | country_uk, germany, france, eire (others = all zeros) | Step 2 found different cancel rates by country |

### Handling missing values (no imputation needed – each gap has a meaning)
| Case | Fill | Reason |
|---|---|---|
| No orders in a window | 0 | They really bought nothing |
| Only one order, so no gap | mean gap = tenure, std = 0 | Tenure is the only interval we know |
| Never cancelled | days_since_cancel = 999 | 0 would mean "cancelled today" – the opposite |
| Product sold < 100 units | product cancel rate = 0 | Too little data to call it risky |

**A leakage trap avoided:** product cancel rates are recomputed **at each cutoff from past data only**.
Using the full-year rates would quietly feed future returns into the features.

## Sanity check (Spearman correlation with the label, training data)
* **Churn** rises with recency (+0.25), longer usual gaps (+0.19) and overdue_ratio (+0.14); it falls
  with frequency (−0.39) and spend (−0.37). This matches intuition.
* **Cancellation** rises with spend (+0.32), past cancellations (+0.27) and frequency (+0.28).
  Big, active customers cancel more – so in Step 4 the cancel model must beat a "rank by spend" baseline.
