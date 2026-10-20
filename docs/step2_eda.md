# Step 2 – Exploratory analysis and process audit

Script: `src/02_eda.py` · Tables: `outputs/step2/` · Charts: `outputs/step2/charts/`

## Methods used and why
| Method | Used for | Why this method |
|---|---|---|
| Group-by aggregation (pandas) | Sizing each leak by product, customer, country, month | The basic tool for "where is the money" |
| Pareto / cumulative share | Is leakage concentrated or spread out? | Concentrated problems can be fixed with targeted policies |
| Rate vs. volume | Product cancel **rate** = cancelled units ÷ sold units (only products with ≥500 units sold) | Big sellers get the most cancellations just by volume; the rate shows which products are actually problematic. The 500-unit floor stops tiny products with 1 sale and 1 return showing as "100%" |
| Keyword classification (regex) | Grouping free-text write-off notes into reasons | The notes are messy free text ("damages?", "wet damaged", "??") |
| Mann-Whitney U test | Do customers who cancel spend more? | Spend is heavily skewed, so a t-test (which assumes normality) is wrong; Mann-Whitney compares ranks instead |
| Chi-square test | Do UK and non-UK invoices get cancelled at different rates? | Standard test for comparing proportions between groups |
| Cohort retention matrix | Do new customers come back? | Separates "customers who joined in March" from "customers who joined in September", so seasonality doesn't hide churn |

## Corrections applied during the analysis
* **Dec 2011 excluded from trends and cohorts** – 9 days of data would fake a collapse.
* **Dec 2010 cohort treated separately** – the data starts that month, so it holds everyone who was
  already a customer, not new customers.

## Findings

### 1. Leak sizing (vs £10.0M product sales)
| Leak | Value | % of sales |
|---|---|---|
| Stock write-offs (at selling price, upper bound) | £365k | 3.6% |
| Customer cancellations | £229k | 2.3% |
| Amazon fees (net) | £222k | 2.2% |
| Manual credits (gross) | £147k | 1.5% |
| Bad debt, bank, CRUK, discounts | £32k | 0.3% |
| *Revenue with no customer ID (visibility, not money lost)* | *£1.51M* | *15.1%* |

### 2. Cancellations are concentrated – so they are targetable
* 18% of products = 80% of cancelled £. Product cancel rate: median 0.6%, but the worst 10% exceed 4.4%.
* 35% of customers ever cancelled, but the **top 10% of cancelling customers = 74% of cancelled £**.
* Cancellers are bigger buyers: median spend £1,544 vs £440 (Mann-Whitney p ≈ 10⁻¹⁸⁴). So cancellation
  risk is tied to valuable wholesale accounts – the business can't just block them.
* Monthly spikes are single accounts: one UK customer = £23k of April's £33k (69%).
* By country: Spain 7.1% and EIRE 4.2% vs UK 2.3%; Netherlands 0.1%. Non-UK invoices are cancelled
  more often (17.4% vs 14.5% of invoices, chi-square p = 0.0003).

### 3. Write-offs: half have no real reason recorded
* 50.5% of write-off value has no reason, "check", "?" or a vague note – a record-keeping process failure.
* Damage is the largest explained reason (31%). Top 20 products = 41% of write-off value.

### 4. Manual credits: no control
* 244 lines, £147k; every one is labelled just "Manual" – no reason code.
* The 10 largest lines = 52% of the value (largest single credit £38,970).
* 28% have no customer attached.

### 5. Missing customer ID gets worse in peak season
* Share of revenue with no customer: about 9% in Sep–Oct, jumping to **22% in Nov** and 27% in Dec 2010.
  Data capture breaks down exactly when volume is highest.
* Only 9 of 38 countries have any missing IDs; the UK accounts for nearly all of it (27% of UK lines).

### 6. Customers: concentrated revenue, weak early retention
* Top 20% of customers = 74% of revenue; top 1% = 30%.
* 35% buy only once. Median gap between repeat orders: 28 days (90th percentile 111 days).
* New customers (Jan–Nov 2011 cohorts): only **~20% return the next month**, vs 37% of established customers.
* Seasonality: Sep–Nov averages £1.19M a month vs £0.65M for Jan–Aug.

## What this means for the models (Step 3)
1. **Churn model** – predict whether a customer makes no purchase in the next 90 days. 90 days fits
   the data: 90% of repeat gaps are under 111 days, so 90 days of silence is unusual.
2. **Cancellation-risk model** – predict whether a customer will cancel in the next 90 days. Because
   cancellers are bigger buyers, the model has to beat a simple "rank by spend" baseline to be useful.
3. **Product and process leaks (write-offs, manual credits, missing IDs)** – handled with rules and
   controls, not ML. They are concentrated and explainable, so a model adds nothing.
