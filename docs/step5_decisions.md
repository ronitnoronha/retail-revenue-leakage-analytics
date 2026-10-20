# Step 5 – From model scores to £ decisions

Script: `src/05_decisions.py` · Outputs: `outputs/step5/` (incl. `customer_action_list.csv`)

All figures are measured on the **out-of-time test** (scored 10 Sep 2011, outcomes to 9 Dec 2011),
so "£ reached" is what would really have happened. Anything the data cannot tell us (campaign save
rate, contact cost, how many cancellations a confirmation step prevents) is an **assumption, shown as
a range**.

## Fixes made during this step
| Issue found | Fix |
|---|---|
| Valuing customers by *last-90-day* spend gave £0 to anyone already silent, unfairly zeroing the recency rule | Value = **typical 90-day spend over the customer's whole history** (spend × 90 ÷ tenure, tenure ≥ 90 days) |
| "P(cancel) × lifetime spend" was a ranking score, not £ | Expected £ = P(cancel) × 5.9% × typical 90-day spend. The 5.9% ("when an account cancels, it cancels ~5.9% of its typical 90-day spend") is learned from **training snapshots only**. Check: predicts £58k vs £69k actual |
| Product "ROTATING SILVER ANGELS" showed a 99% cancel rate: 9,360 units cancelled at £0.03 each for an order placed before the data starts | Product review uses **actual cancelled £** and a £-based rate, not units × typical price |

## A. Retention targeting
**Decision:** a retention team can contact a limited number of customers. Who?
**Method:** rank by **expected £ at risk = P(churn) × typical 90-day spend**, and compare with rules.

Lost revenue in the test window (churners' typical 90-day spend): **£466k from 1,064 churners**.

| Contact budget | Model: expected £ at risk | Churn probability only | Biggest spenders | Longest silence | Random |
|---|---|---|---|---|---|
| Top 5% (139) | **29.6%** | 10.4% | 20.6% | 4.0% | 5.2% |
| Top 10% (278) | **38.2%** (£178k) | 12.7% | 28.8% | 6.9% (£32k) | 11.3% |
| Top 20% (556) | **53.0%** | 20.4% | 38.8% | 21.5% | 19.3% |

* At a 10% budget the model reaches **5.5× the at-risk £ of the recency rule** and 1.3× the
  "biggest spenders" rule.
* Probability alone is a poor targeting tool: it finds many churners (181 at 10%) but mostly small
  ones. **Combining probability with value is what makes the model useful.**

**Campaign economics (90 days, top 10%, assumptions):** net £ = save rate × £ reached − cost × 278.

| Save rate \ cost per contact | £10 | £25 | £50 |
|---|---|---|---|
| 5% | £6.1k | £2.0k | −£5.0k |
| 10% | £15.0k | £10.9k | £3.9k |
| 20% | £32.8k | £28.7k | £21.7k |

The model-targeted campaign is positive in 8 of 9 scenarios; the recency-rule campaign is negative
in 6 of 9. Real save rates need an A/B test (hold out a random control group).

## B. Cancellation policy
**Decision:** add an order-confirmation step for high-risk accounts. Which ones?

| Flag top 5% (139 accounts) | Cancelled £ covered | Next-90-day revenue touched |
|---|---|---|
| Model: expected £ loss | 55.9% | 46.0% |
| Rule: biggest spenders | 56.0% | 46.8% |
| Model: probability only | 47.1% | 40.1% |
| Rule: past cancellers by £ | 46.4% | **28.7%** |

**Honest reading:**
* In £ terms the model **ties** the "biggest spenders" rule – cancelled £ is concentrated in the
  biggest accounts, and spend already captures that.
* Flagging those accounts puts friction on **46% of revenue**. That's too much for an automatic
  confirmation step.
* **Recommendation:** use account-manager review (not friction) for the top accounts, and an
  automatic confirmation step only for past cancellers, which covers 46% of cancelled £ while
  touching 29% of revenue.
* Savings if confirmation prevents 20–50% of flagged cancellations: **£7.7k–£19.3k per 90 days**.

## C. Process controls (rules, not models)
| Leak | Finding | Control | Impact |
|---|---|---|---|
| Manual credits (£147k) | No reason codes | Second approval for credits ≥ £500 | 60 of 244 credits (25%) cover **86%** of manual credit £ – low workload, high coverage |
| Stock write-offs (£365k at selling price) | **£184k (50.5%)** has no usable reason | Mandatory reason codes; packaging review for damage (£114k) | Makes half the write-off value explainable |
| Product cancellations | **157 of 1,628 products** (≥5% of their sales £ cancelled) = **50% of product cancel £** | Quality / listing review for those 157 products | Focuses review on 10% of products |
| Missing customer ID | **£1.51M** of revenue unattributed; **£316k in Nov 2011** alone | Make customer ID mandatory at checkout, especially in peak season | Restores visibility for retention and targeting |

## D. Customer action list (`customer_action_list.csv`)
| Action | Rule | Customers | Actual churn | Actual cancel |
|---|---|---|---|---|
| 1. Account review | Top 5% cancel £ risk AND top 10% churn £ risk | 21 | 24% | 48% |
| 2. Confirm orders | Top 5% cancel £ risk | 119 | 2% | 62% |
| 3. Retention contact | Top 10% churn £ risk | 258 | 41% | 17% |
| 4. Win-back email | P(churn) ≥ 50% | 993 | 60% | 7% |
| 5. Business as usual | everyone else | 1,392 | 26% | 24% |

The action groups separate cleanly on real outcomes: "confirm orders" accounts cancelled at 62%
(vs 19% overall) and "win-back" customers churned at 60% (vs 38% overall).
