# Step 8 – Customer segmentation for targeted marketing

Script: `src/08_segments.py` (run as `python src/08_segments.py 5`) · Outputs: `outputs/step8/`

## Question
The churn model answers *"who is about to leave?"*. Segmentation answers *"what kinds of customers do we
have, and how should we treat each kind?"*: one message per group instead of one for everyone.

## Design: segment on the past, validate on the future
* **Customers:** the 2,783 active on 10 Sep 2011 (bought in the previous 180 days) – the same snapshot the
  churn model scored.
* **Features:** behaviour known on 10 Sep only (from Step 3).
* **Validation:** what each segment *actually did* from 10 Sep to 9 Dec 2011. A segmentation that
  doesn't predict different behaviour is just a pretty chart.

## Method
| Step | What | Why |
|---|---|---|
| Features (10) | recency, orders, spend, order value, tenure, units per order, distinct products, unit price, spend trend, cancel rate | One feature per behaviour, so no behaviour gets double weight |
| Transform | `log1p`, then standardise (mean 0, sd 1) | Clustering uses distances. Without the log, a few £100k+ customers would dominate; without scaling, £ would outweigh days |
| Algorithms compared | K-Means, Gaussian Mixture, Agglomerative (Ward), for k = 2–10 | Three different ideas of a "group": nearest centre, overlapping probability blobs, bottom-up merging |
| Metrics | Silhouette (higher better), Davies-Bouldin (lower better), smallest segment size, elbow (inertia), BIC | No single metric is enough; plus business usefulness |
| Stability | Refit with 10 other seeds and on 20 random 80% subsamples; Adjusted Rand Index vs final | Segments must not change every time you run it |

### How the algorithms work
* **K-Means:** place k centres, assign each customer to the nearest, move each centre to the mean of its
  customers, repeat until stable. Fast, simple, every customer in exactly one group.
* **Gaussian Mixture:** assumes each segment is an elliptical "cloud" and gives each customer a
  probability of belonging to each. More flexible, but fragile here.
* **Agglomerative (Ward):** start with every customer alone, repeatedly merge the two groups whose
  merge increases within-group spread the least.

## Choosing k
| | k=3 | k=4 | **k=5** | k=6 |
|---|---|---|---|---|
| Silhouette (K-Means) | 0.230 | 0.183 | 0.191 | 0.192 |
| Davies-Bouldin (K-Means) | 1.50 | 1.55 | **1.37** | 1.39 |

* **K-Means beat the other two algorithms at almost every k.** The Gaussian Mixture fell apart beyond
  k=4: it started carving out tiny spikes in the data, which is why its BIC kept improving.
* **k=5:** best Davies-Bouldin, and every segment maps to a distinct marketing action. k=3 scores a
  higher silhouette but merges new customers with lapsing ones, which need opposite treatment.
* **Silhouette is low everywhere (≤ 0.23).** Customer behaviour is a continuum, not separate clumps.
  Segments here are useful *divisions* of that continuum, not natural species. Say this in interviews.
* **Stability:** mean ARI 0.92 across seeds and 0.94 across 80% subsamples, so the same segments come
  back almost every time (a few runs split them differently: minimum 0.56–0.62). Agreement with Ward
  clustering is moderate (0.41): different algorithms draw the boundaries in different places, as
  expected on a continuum.

## The five segments (as of 10 Sep 2011)
| Segment | Customers | Spend share | Median orders | Median spend | Days since order | Actual churn (next 90 days) | Marketing action |
|---|---|---|---|---|---|---|---|
| **Loyal wholesale core** | 594 (21%) | **67%** | 8 | £2,690 | 12 | **11%** | Protect: account manager, early access, volume pricing |
| **Regular mid-tier** | 995 (36%) | 22% | 3 | £870 | 80 | 36% | Re-engage: reorder reminders timed to their usual gap; main retention-campaign pool |
| **New customers** | 485 (17%) | 5% | 1 | £375 | 31 | 50% | Onboard: second-order offer within ~30 days |
| **Lapsing small buyers** | 665 (24%) | 4% | 1 | £229 | 101 | **57%** | Cheap automated win-back email; no phone calls |
| **Serial cancellers** | 44 (2%) | 2% | 2 | £538 | 105 | 52% | Order confirmation or deposit; find out why they cancel |

**Validation:** actual churn ranges from 11% to 57% across segments (overall 38%); chi-square
p ≈ 10⁻⁶⁹. The segments separate real future behaviour, not just past behaviour.

**Notable findings**
* **21% of customers bring 67% of spend**, and they almost never leave (11% churn), but they cancel the
  most (39% cancelled something in the next 90 days) because they order the most. Friction at checkout
  would hit the business's best customers – handle them through account managers.
* **Half of new customers never come back within 90 days.** The second order is the critical moment;
  the median gap between repeat orders is 28 days, so the offer should land within a month.
* **Serial cancellers** cancel 49% of what they buy. They are only 44 customers, which makes a targeted
  confirmation step cheap.
* **Regular mid-tier holds the most churn £ at risk (£186k of expected loss).** That is where a
  retention budget does the most good.

## How it fits with the churn model
Segments decide **what message** a customer gets; the churn score decides **who gets contacted first
within each segment**. Example: a Regular mid-tier customer with high churn probability gets a call;
one with low probability gets the reorder reminder only.
