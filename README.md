# Retail Revenue Leakage & Customer Risk Analytics

Where does a UK online retailer lose money between order and cash, and what should it do about it?

**Data:** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), 2010–2011 sheet –
541,910 transactions, £10.0M product sales, 4,333 customers, 38 countries.

## Headline results
* **Up to ~£1M (9.9% of sales) of value leakage**, mostly from internal processes: stock write-offs
  (£365k at selling price, half with no recorded reason), customer cancellations (£229k), manual credits
  (£147k, no reason codes) and fees.
* **Raw data overstated cancellations 3.6×** (8.4% → 2.3% of sales) because of keying errors and fee reversals.
* **7 churn models from 6 families** compared on an out-of-time test: logistic regression tied for best
  (ROC-AUC 0.748 vs 0.656 for a recency rule) and was chosen for explainability.
* **Value-based targeting:** contacting the top 10% of customers by expected £ at risk reaches **38% of
  at-risk revenue – 5.5× a recency rule**.
* **5 customer segments** (K-Means, chosen over Gaussian Mixture and Ward): 21% of customers bring 67% of
  spend and churn at 11%; half of new customers don't return within 90 days. Segments were validated on
  the next 90 days (churn 11%–57% by segment).
* **Controls:** a £500 second-approval rule covers 86% of manual-credit value; 157 products (10%) cause
  50% of product-cancellation £.

See [`docs/recommendation_memo.md`](docs/recommendation_memo.md) for the one-page client memo.

## Pipeline
| Step | Script | Write-up |
|---|---|---|
| 1. Audit & cleaning | `src/01_clean.py` | [`docs/step1_cleaning.md`](docs/step1_cleaning.md) |
| 2. Exploratory analysis & process audit | `src/02_eda.py` | [`docs/step2_eda.md`](docs/step2_eda.md) |
| 3. Time-based feature engineering | `src/03_features.py` | [`docs/step3_features.md`](docs/step3_features.md) |
| 4. Model comparison (7 models) | `src/04_models.py`, `src/bgnbd.py` | [`docs/step4_model_guide.md`](docs/step4_model_guide.md) |
| 5. Decisions & policy simulation | `src/05_decisions.py` | [`docs/step5_decisions.md`](docs/step5_decisions.md) |
| 6. Power BI export | `src/06_export_powerbi.py` | [`powerbi/README.md`](powerbi/README.md) |
| 8. Customer segmentation (K-Means, 5 segments) | `src/08_segments.py` | [`docs/step8_segmentation.md`](docs/step8_segmentation.md) |
| 7. Online dashboard (4 tabs) | `dashboard/build_data.py`, `dashboard/index.html` | data in `dashboard/data/` |

## Models compared (churn, out-of-time test 10 Sep – 9 Dec 2011)
| Model | Family | Test ROC-AUC |
|---|---|---|
| Logistic Regression | Linear | **0.748** |
| Cox Proportional Hazards | Survival | **0.748** |
| LightGBM | Gradient boosting | 0.746 |
| Random Survival Forest | Survival + ensemble | 0.745 |
| Random Forest | Tree ensemble | 0.744 |
| MLP | Neural network | 0.743 |
| BG/NBD (implemented from Fader, Hardie & Lee 2005) | Probabilistic | 0.734 |
| Recency rule | Baseline | 0.656 |

## Run it
```bash
pip install -r requirements.txt
# put online_retail_II_2010_2011.csv (the 2010-2011 sheet saved as CSV) in data/raw/
python src/01_clean.py
python src/02_eda.py
python src/03_features.py
python src/04_models.py      # ~13 min on 4 cores (hyperparameter search)
python src/05_decisions.py
python src/06_export_powerbi.py
```
