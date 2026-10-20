# Resume entry and interview prep – Accenture I&F Decision Science Practitioner Analyst

## 1. Resume changes
* **Replace** "Diabetes Prediction System" with this project, and list this project **first** – it is
  the one that matches the JD ("insights about the effectiveness of business processes, procedures
  and policies").
* Keep the Credit-Card Fraud project (strong ML) and SAGE (shows GenAI range).
* Only include the Power BI wording **after you have built the dashboard** (Step 6 guide).

### Version A – 3 bullets (recommended)
**Retail Revenue Leakage & Customer Risk Analytics** | Python, scikit-learn, Power BI | Oct 2026
* Audited 541K transactions (£10M sales) for a UK online retailer, sizing up to £1M of value leakage;
  found 50% of write-off value had no recorded reason and that a £500 approval rule covers 86% of manual-credit £.
* Benchmarked 7 churn models across 6 families (logistic, survival, BG/NBD built from scratch, boosting,
  neural net) on an out-of-time test; chose logistic regression as tied-best (AUC 0.75 vs 0.66 baseline).
* Ranked customers by revenue at risk, reaching 38% of at-risk revenue by contacting 10% of customers
  (5.5× a recency rule); delivered a Power BI dashboard, customer action list and 3 costed recommendations.

### Version B – 2 bullets (matches your current resume layout)
**Retail Revenue Leakage & Customer Risk Analytics** | Python, scikit-learn, Power BI | Oct 2026
* Audited 541K transactions (£10M sales), sizing up to £1M of leakage across write-offs, cancellations and manual
  credits, and designed controls such as a £500 approval rule covering 86% of manual-credit value.
* Benchmarked 7 churn models across 6 families on an out-of-time test (best AUC 0.75 vs 0.66 baseline) and
  ranked customers by revenue at risk, reaching 38% of it by contacting 10% of customers (5.5× a recency rule).

### Skills line – suggested addition
Statistics: hypothesis testing, survival analysis, probabilistic customer models (BG/NBD)

### Every number, and where it comes from
| Claim | Source |
|---|---|
| 541K transactions, £10M sales | Step 1: 541,910 raw rows; £10,000,733 product sales |
| Up to £1M leakage | Step 6 check total £994,143 – includes write-offs at selling price (upper bound) and Amazon fees |
| 50% of write-off value no reason | Step 2: 50.5% (£184k) |
| £500 rule covers 86% of manual-credit £ | Step 5: 60 of 244 credits, 85.8% of £147k |
| 7 models, 6 families | Step 4: LR, RF, LightGBM, MLP, Cox, RSF, BG/NBD |
| AUC 0.75 vs 0.66 | Step 4 test: 0.748 vs 0.656 |
| 38% of at-risk revenue, 10% of customers, 5.5× | Step 5: £178k of £466k vs £32k for the recency rule |

---

## 2. The 60-second pitch ("Tell me about a project")
> "I took a year of transactions from a UK online retailer – about 540,000 rows, £10 million in sales –
> and asked where the business was losing money between order and cash. The first thing I learned was
> not to trust the raw numbers: cancellations looked like 8% of sales, but two keying errors – one was
> 80,995 units ordered and reversed twelve minutes later – plus fee reversals were inflating that. The
> real figure was 2.3%.
>
> Once cleaned, the biggest losses turned out to be internal processes: half of stock write-off value
> had no reason recorded, and manual credits had no controls. For those I recommended simple rules –
> for example, a second approval above £500 covers 86% of manual-credit value while touching only a
> quarter of the credits.
>
> On the customer side, I built churn models – seven, from six families, including survival analysis
> and a BG/NBD model I implemented from the original paper – and tested them on a later period they'd
> never seen. Logistic regression tied for best, so I chose it because it's explainable. The real value
> came from combining the churn probability with customer value: contacting just 10% of customers
> reaches 38% of the revenue at risk, five and a half times a simple recency rule."

---

## 3. Likely questions and strong answers

### About the project
**Q: Why this dataset and this problem?**
The role is about evaluating business processes and policies. Leakage is exactly that: it sits in
processes (write-offs, credits, data capture), and the fix is a policy, not just a model.

**Q: What was the hardest part?**
Data quality. Keying errors, fees mixed into sales, write-offs disguised as negative sales, 25% of lines
with no customer, a partial final month. Each needed a rule, and every rule changed the headline numbers.

**Q: How did you handle the 25% of rows with no Customer ID?**
I didn't impute them – you can't guess who bought something. I kept them in revenue totals, excluded
them from customer models, and reported the missing rate itself as a finding: it peaks at 22% of
revenue in November, exactly when volume is highest.

**Q: Why didn't you remove outliers?**
This is a wholesaler, so huge orders are real business. I only removed proven errors: a purchase
reversed within 24 hours with the identical quantity.

**Q: Why did logistic regression win over XGBoost / LightGBM / a neural network?**
All seven models finished within 0.015 AUC of each other. When very different algorithms converge,
the limit is the information in the features, not the algorithm. The tuned tree models even chose
shallow, heavily regularised settings – a sign the relationship is close to linear once spend is
log-transformed. So I picked the simplest, fastest and most explainable one.

**Q: Why didn't you use deep learning?**
I tested a neural network (MLP). On ~9,000 rows of tabular data it scored 0.743 vs 0.748 for logistic
regression – in line with research showing tree and linear models usually beat neural nets on small
tabular data. Deep learning pays off with images, text or very large datasets.

**Q: What is data leakage and how did you prevent it?**
Using information in training that wouldn't be available at prediction time. I built features at
fixed snapshot dates using only data before each date, predicted the 90 days after, tested on a later
period than any training label, and recomputed even product-level cancel rates from past data only.
The code has assertions that fail if any feature touches future data.

**Q: Why GroupKFold?**
The same customer appears in several snapshots. Ordinary k-fold could put them in both training and
validation, inflating scores. Grouping by customer keeps each customer on one side.

**Q: Why 90 days for churn?**
From the data: 90% of gaps between repeat orders were under 111 days, so 90 days of silence is unusual.

**Q: Explain BG/NBD simply.**
Each customer buys at their own rate while they're "alive", and after each purchase has some chance of
quietly leaving for good. From just three numbers per customer – how many repeat purchases, when the
last one was, and how long we've known them – it estimates the probability they're still active and
how much they'll buy. It needs no labels.

**Q: What is censoring in survival analysis?**
If a customer hasn't bought again yet, we don't know they've left – only that they haven't bought *so
far*. Survival models treat that as incomplete information rather than a "no", which is statistically
correct, and they predict *when*, not just *whether*.

**Q: ROC-AUC vs PR-AUC?**
ROC-AUC: how often the model ranks a random positive above a random negative. PR-AUC focuses on the
positive class and is more informative when positives are rare – cancellation is 19% of customers.

**Q: What would you do differently or next?**
(1) An A/B test with a hold-out group to measure the real save rate of retention contacts.
(2) Uplift modelling, to target customers who change behaviour *because* of the contact – that needs
campaign data. (3) Cost-price data to value write-offs properly. (4) Model seasonality explicitly.

**Q: Where did a model NOT help?**
Cancellation targeting: in £ terms the model tied a "biggest spenders" rule, and flagging those
accounts would add friction to 46% of revenue. So I recommended a simpler policy – confirmation only
for past cancellers – plus account-manager reviews for large accounts.

### Business / consulting judgement
**Q: How would you present this to a client who isn't technical?**
Lead with the money and the action: "Up to £1M leaks out; half the write-offs have no reason; one
approval rule covers 86% of manual credits; calling 10% of customers protects 38% of at-risk revenue."
Models go in the appendix.

**Q: Your numbers rely on assumptions – how do you defend them?**
I separate measured facts from assumptions and show assumptions as ranges. The retention campaign is
positive in 8 of 9 scenarios, so the recommendation doesn't depend on one guess – and the first step is
an A/B test to replace the assumption with a measurement.

**Q: Is "£1M of leakage" all money lost to mistakes?**
No – it's an upper bound. Write-offs are valued at selling price, and Amazon fees are a channel cost,
not an error. Customer cancellations (£229k) and manual credits (£147k) are the firmer figures.

### JD and behavioural
**Q: The JD asks for Python and basic data science. What's your level?**
Point to the pipeline: pandas cleaning and feature engineering, scikit-learn pipelines with tuning and
grouped CV, and two statistical models implemented from scratch with SciPy optimisation.

**Q: This role involves lower-complexity tasks with moderate instructions. Are you OK with that?**
Yes – most real value comes from careful basics. In this project the biggest correction (8.4% → 2.3%)
came from checking the data, not from a model, and the winning model was the simplest one.

**Q: Tell me about working in a team / commitment to quality.**
Use the DotCom Club e-waste drive (20+ member team, 1,000+ students, 454 kg) for teamwork; use the
keying-error catch and the reconciliation checks in this project for quality.
