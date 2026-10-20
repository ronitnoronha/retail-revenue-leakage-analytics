# Recommendation memo: closing revenue leakage at a UK online retailer

**To:** Head of Operations · **From:** Ronit Noronha, Analytics · **Data:** 541,910 transactions, Dec 2010 – Dec 2011

## Situation
The business sold **£10.0M** of products to 4,333 identifiable customers in 38 countries. Up to
**~£1M (9.9% of sales)** of value leaks out between order and cash. Most of it comes from **internal
processes** rather than customers, and much of it is invisible in current reporting.

| Leak | Value | Root cause |
|---|---|---|
| Stock write-offs (at selling price, upper bound) | £365k | Half the value (£184k) has no recorded reason |
| Customer cancellations | £229k (2.3% of sales) | Concentrated: 10% of products = 50% of product cancel £ |
| Amazon fees (channel cost) | £222k | Not an error – a margin question for the channel |
| Manual credits | £147k | No reason codes; 10 credits = 52% of the value |
| Other finance charges | £32k | Bad debt, bank charges, commission |
| *Sales with no customer ID* | *£1.51M (15%)* | *Not lost, but untraceable – rises to 22% in November* |

Note: raw data overstated cancellations at 8.4% of sales. Two keying errors (e.g. 80,995 units
ordered and reversed 12 minutes later) and fee reversals were removed; the true figure is 2.3%.

## Recommendations

**1. Put basic controls on internal adjustments (no technology needed)**
* Two-person approval for manual credits **≥ £500**: only **60 of 244 credits (25%)** would need a
  second sign-off, yet they cover **86% (£126k)** of manual-credit value.
* Mandatory reason codes for stock write-offs, plus a packaging review for the 20 products behind
  41% of write-off value (damage alone = £114k).
* Make customer ID mandatory at checkout before peak season: £316k of November revenue had no
  customer attached, so those buyers can't be retained or analysed.

**2. Target retention by revenue at risk, not by silence**
* A churn model scores every customer's risk of not ordering in the next 90 days; multiplying risk
  by the customer's typical spend gives **£ at risk**.
* Tested on a period the model never saw (Sep–Dec 2011): contacting the **top 10% of customers
  reaches 38% of at-risk revenue (£178k)** – **5.5×** the "longest silence" rule (£32k).
* Expected net gain about **£11k per quarter** at a 10% save rate and £25 per contact; across save
  rates of 5–20% and costs of £10–£50 it ranges from −£5k to +£33k, positive in 8 of 9 scenarios.

**3. Cut cancellations at the source**
* Review listings, descriptions and quality for the **157 products** where ≥5% of sales £ is
  cancelled (**50% of product cancellation £**).
* Add an order-confirmation step only for **past cancellers** (5% of accounts): covers 46% of
  cancelled £ while touching 29% of revenue. Saves **£6k–£16k per quarter** if it prevents 20–50% of
  their cancellations. Largest accounts get an account-manager review instead of added friction.

## Evidence and method
* Customer-level features built at fixed snapshot dates (no future information); 7 models from 6
  families compared on an out-of-time test with customer-grouped cross-validation.
* Logistic regression chosen: tied-best accuracy (ROC-AUC 0.75 vs 0.66 for the simple rule), fastest
  and fully explainable. Complex models (gradient boosting, neural network, survival forest) added
  no accuracy.

## Assumptions and next steps
* Save rates, contact costs and prevention rates are assumptions, shown as ranges. **Run an A/B test**
  (random hold-out group) on the first retention campaign to measure the real save rate.
* Write-offs are valued at selling price (upper bound); cost-price data would give the true loss.
* Refresh the scores monthly; monitor model accuracy each quarter, especially after peak season.
