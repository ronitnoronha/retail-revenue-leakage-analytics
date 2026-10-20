# Step 4 – How each model works

Script: `src/04_models.py` (BG/NBD and Gamma-Gamma implemented in `src/bgnbd.py`)

Seven models from six families, plus a simple rule as the baseline. Each section answers: **what it
does, how it works, what it's good and bad at, and how we set it up.**

---

## 0. Baseline rules (no model)
* **Churn:** rank customers by *recency* (days since last order). The longer they've been silent,
  the more likely they've left.
* **Cancellation:** rank customers by *total spend*. Big buyers cancel more (Step 2).

**Why:** a model is only worth building if it beats what an analyst could do in Excel in five minutes.
Every result is read against these.

---

## 1. Logistic Regression – *linear family*
**What it does:** estimates the probability of churn as a weighted sum of the features, squeezed
into 0–1 by the sigmoid curve: `p = 1 / (1 + e^-(b0 + b1·x1 + b2·x2 + …))`.

**How it learns:** finds the weights that make the observed outcomes most likely (maximum
likelihood). Regularisation (`C`) shrinks the weights so it doesn't over-fit noise.

**Strengths:** fully explainable – each weight says how much a feature pushes risk up or down;
fast; well-calibrated probabilities. **Weaknesses:** only captures straight-line effects unless you
build interactions by hand (e.g. "high spend AND long silence").

**Our setup:** features log-transformed (to tame skew) and standardised (so weights are comparable);
`C` tuned from 0.01 to 10.

---

## 2. Random Forest – *tree ensemble (bagging)*
**What it does:** builds hundreds of decision trees and averages their votes.

**How it works:** each tree is grown on a random bootstrap sample of the rows and, at each split,
only considers a random subset of features. A single tree asks yes/no questions ("recency > 45 days?
→ frequency < 3?") and over-fits easily; averaging many *different* trees cancels out their
individual mistakes.

**Strengths:** captures non-linear effects and interactions automatically; robust; needs no scaling.
**Weaknesses:** less interpretable; probabilities can be poorly calibrated; large models.

**Our setup:** 400 trees; tuned tree depth, minimum leaf size and features per split.

---

## 3. LightGBM – *gradient boosting*
**What it does:** builds trees **one after another**, each new tree correcting the errors of the
ones before it.

**How it works:** start with a basic prediction; compute how wrong it is for every customer; fit a
small tree to those errors (the "gradient"); add a fraction of it (learning rate); repeat hundreds
of times. LightGBM grows trees leaf-by-leaf and bins feature values into histograms, which makes it
much faster than classic boosting. Same family as XGBoost, different engine.

**Strengths:** usually the most accurate on tabular data. **Weaknesses:** many settings to tune;
can over-fit small data; less transparent.

**Our setup:** 400 trees; tuned learning rate, leaves per tree, minimum samples per leaf, row/column
sampling and L2 regularisation.

---

## 4. MLP Neural Network – *neural / deep learning*
**What it does:** passes the features through layers of "neurons"; each neuron computes a weighted
sum and applies a non-linear activation (ReLU), so stacked layers can represent complex patterns.

**How it learns:** backpropagation – predict, measure the error, push the error backwards through
the layers to adjust every weight a little (Adam optimiser), and repeat over many passes.
*Early stopping* holds back 10% of training data and stops when that score stops improving.

**Strengths:** very flexible; the best tool for images, text and sequences.
**Weaknesses:** on small tabular data it usually loses to tree models; needs scaled inputs; a black
box; results vary with the random seed.

**Our setup:** tuned architecture (1–3 hidden layers, 32–128 neurons), L2 penalty `alpha` and
learning rate.

---

## 5. Cox Proportional Hazards – *survival analysis*
**What it does:** instead of "will they churn?", it models **time until the next purchase**.

**How it works:** every customer has a *hazard* – their instantaneous chance of buying at any moment.
Cox assumes `hazard(t) = baseline_hazard(t) × e^(b1·x1 + b2·x2 + …)`: the features scale a shared
baseline curve up or down ("proportional" hazards). From this we get each customer's **survival
curve** S(t) = probability of *no* purchase by day t. Churn probability = S(90).

**Why survival matters – censoring:** a customer who hasn't bought yet *might still buy tomorrow*.
Survival models treat "not yet" as incomplete information (censored) rather than a final "no",
which is the statistically correct way to handle it. They also give answers for any horizon
(30, 60, 90 days) from one model.

**Strengths:** principled; interpretable (hazard ratios); answers "when", not just "if".
**Weaknesses:** linear in the features, and assumes effects are constant over time.

**Our setup:** ridge penalty (alpha = 1) for stability; same log + standardised features as
Logistic Regression.

---

## 6. Random Survival Forest – *survival + tree ensemble*
**What it does:** a Random Forest for survival data. Each tree splits customers into groups with
the most different survival curves (log-rank test); the forest averages the curves.

**Strengths:** non-linear survival curves, no proportional-hazards assumption.
**Weaknesses:** slow; memory-heavy.

**Our setup:** 200 trees, minimum 20 customers per leaf, √features per split.

---

## 7. BG/NBD (+ Gamma-Gamma) – *probabilistic customer-base model*
**What it does:** the marketing-science standard for "is this customer still active?" in businesses
without subscriptions, where customers leave silently.

**How it works – the story the model tells:**
1. While "alive", each customer buys at their own steady rate λ (a Poisson process). Rates differ
   between customers, following a Gamma distribution (parameters r, α).
2. After every purchase, each customer may quietly leave forever with their own probability p.
   Those probabilities follow a Beta distribution (parameters a, b).

The four parameters are fitted by maximum likelihood using only three numbers per customer:
**x** (repeat purchases), **t_x** (when the last purchase was) and **T** (how long we've known them).
For each customer it then gives **P(still alive)** and **expected purchases in the next 90 days**.

**Gamma-Gamma** adds money: it estimates each customer's typical order value, shrinking customers
with few orders toward the average. Expected purchases × expected value = **customer lifetime value**.

**Strengths:** needs no labels; minimal data; produces CLV in £; grounded in behavioural theory.
**Weaknesses:** uses only frequency and recency (ignores basket, cancellations, country); assumes a
steady buying rate, so it **can't anticipate seasonal peaks**.

**Implementation note:** the `lifetimes` library doesn't install on this Python version, so both
models were implemented from the original papers (Fader, Hardie & Lee 2005) with SciPy.
With under a year of history the dropout parameters (a, b) were weakly identified – the likelihood
was flat in that direction – so a small penalty on a and b only was added for stability.

---

## How they're compared
| Metric | Question it answers | Why we need it |
|---|---|---|
| **ROC-AUC** | If I pick one churner and one non-churner at random, how often does the model rank the churner higher? 0.5 = coin flip, 1.0 = perfect | Overall ranking quality |
| **PR-AUC** (average precision) | How precise are the model's top picks across all thresholds? | Better than ROC-AUC when positives are rare (cancellation: 19%) |
| **Brier score** | Mean squared error of the probabilities (lower is better) | Step 5 multiplies probabilities by £, so they must be accurate, not just well-ranked |
| **Top-20% capture** | If the team can only contact the riskiest 20% of customers, what share of actual churners/cancellers do they reach? | The operational question a manager asks |
| **Top-20% £ capture** (cancellation) | Same, weighted by cancelled £ | Catching a £5k canceller matters more than a £5 one |
| **CV ± sd** | Average and spread across 5 customer-grouped folds | Stability – a model that's great on one fold and poor on another isn't trustworthy |
| **Fit time** | Training + tuning time | Practicality |

**Validation design:** tuning uses only the training snapshots with **GroupKFold by customer**; the
test snapshot (10 Sep → 9 Dec 2011) is used once, at the end. Test results are the honest numbers.

---

# Results (out-of-time test: 10 Sep – 9 Dec 2011)

## Churn (base rate 38.2%, 2,783 customers)
| Model | CV ROC-AUC (± sd) | Test ROC-AUC | Test PR-AUC | Brier | Top-20% capture | Fit time |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.742 ± 0.012 | **0.748** | **0.607** | **0.196** | 33.8% | 3s |
| Cox Proportional Hazards | 0.741 ± 0.014 | **0.748** | 0.605 | **0.196** | 33.6% | 2s |
| LightGBM | 0.738 ± 0.013 | 0.746 | 0.598 | 0.197 | 33.6% | 41s |
| Random Survival Forest | 0.732 ± 0.013 | 0.745 | 0.592 | 0.197 | 33.7% | 78s |
| Random Forest | 0.739 ± 0.013 | 0.744 | 0.596 | 0.198 | 33.6% | 136s |
| MLP Neural Net | 0.739 ± 0.011 | 0.743 | 0.598 | 0.199 | 33.3% | 44s |
| BG/NBD (no labels, 4 inputs) | 0.718 ± 0.014 | 0.734 | 0.586 | 0.212 | 32.7% | 0.1s |
| Baseline: recency rule | 0.643 ± 0.021 | 0.656 | 0.528 | – | 30.7% | – |

## Cancellation (base rate 18.8%)
| Model | CV ROC-AUC (± sd) | Test ROC-AUC | Test PR-AUC | Brier | Top-20% capture | Top-20% £ capture |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.764 ± 0.016 | **0.767** | **0.477** | 0.128 | **49.5%** | 68.8% |
| MLP Neural Net | 0.761 ± 0.019 | 0.765 | 0.471 | 0.128 | 47.8% | 67.7% |
| LightGBM | 0.766 ± 0.015 | 0.762 | 0.462 | 0.128 | 48.4% | 67.4% |
| Random Forest | 0.766 ± 0.015 | 0.763 | 0.460 | 0.128 | 46.8% | 66.9% |
| Baseline: spend rule | 0.741 ± 0.019 | 0.729 | 0.404 | – | 42.4% | **73.6%** |

## What the results mean
1. **Every model clearly beats the simple rules** (churn AUC 0.75 vs 0.66; cancellation PR-AUC
   0.48 vs 0.40). Modelling adds real value.
2. **All seven churn models land within 0.015 AUC of each other.** When very different algorithms
   converge, the limit is the information in the data, not the algorithm. More complex models found
   no extra pattern.
3. **Logistic Regression wins (tied with Cox)** – the simplest, fastest, most explainable model.
   Tree models and the neural net picked heavily regularised settings (shallow trees, big leaves),
   another sign the true relationship is close to linear on log-scaled features.
4. **The neural network did not beat the trees or the linear model**, as expected for ~9k rows of
   tabular data.
5. **BG/NBD came within 0.014 AUC of the best** using only 4 numbers per customer and no labels.
   Its weakness is seasonality: its 90-day CLV forecast was £1.48M vs £2.51M actual, because it
   assumes steady buying rates and the test window is the Sep–Nov peak. Its ranking is still useful
   (Spearman 0.53).
6. **Cancellation trade-off:** the model finds more cancelling *customers* in its top 20% (49.5% vs
   42.4%), but the spend rule captures more cancelled *£* (73.6% vs 68.8%), because the largest
   cancellations come from the largest buyers. Step 5 combines both: expected loss =
   P(cancel) × customer value.
7. **Stable:** cross-validation spread is ±0.01–0.02 for every model, and test scores match CV.

## What drives the predictions (Logistic Regression)
* **Churn** – frequency of orders and orders in the last 90 days matter most, then spend, product
  range and tenure (all reduce churn). Longer usual gaps between orders and being "overdue" raise it.
* **Cancellation** – past cancellations dominate (shuffling that feature drops PR-AUC by 0.20), then
  order size (units per order), product range and spend. Wholesale-style buyers cancel more.
