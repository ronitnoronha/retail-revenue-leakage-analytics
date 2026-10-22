"""
Collect the small summary tables the Streamlit app reads into app/data/.
Run from the project root after Steps 1-8 and dashboard/build_data.py:
    python app/build_app_data.py
"""
import os
import shutil
import pandas as pd

OUT = "app/data/"
os.makedirs(OUT, exist_ok=True)

# Summary tables already built for the online dashboard
for f in os.listdir("dashboard/data"):
    shutil.copy("dashboard/data/" + f, OUT + f)

# Segment profiles (Step 8)
prof = pd.read_csv("outputs/step8/segment_profiles.csv")
prof.rename(columns={prof.columns[0]: "segment"}).to_csv(OUT + "segment_profiles.csv", index=False)

# One row per scored customer: segment, scores, recommended action, and what actually happened
seg = pd.read_csv("outputs/step8/customer_segments.csv", dtype={"CustomerID": str})
act = pd.read_csv("outputs/step5/customer_action_list.csv", dtype={"CustomerID": str})
sales = pd.read_parquet("data/processed/sales.parquet")
country = sales[sales.HasCustomer].groupby("CustomerID")["Country"].agg(lambda c: c.mode().iloc[0])
c = seg.merge(act[["CustomerID", "action", "p_cancel", "value_90d", "churn_value_at_risk", "cancel_exp_loss"]],
              on="CustomerID", how="left")
c["country"] = c["CustomerID"].map(country)
c["action"] = c["action"].str.replace(r"^\d\. ", "", regex=True)
cols = {"CustomerID": "customer_id", "segment": "segment", "country": "country", "action": "action",
        "recency_days": "days_since_last_order", "frequency": "orders", "monetary": "total_spend_gbp",
        "avg_order_value": "avg_order_value_gbp", "avg_units_per_order": "units_per_order",
        "n_products": "distinct_products", "cancel_rate": "past_cancel_rate", "p_churn": "churn_probability",
        "p_cancel": "cancel_probability", "value_90d": "typical_90d_spend_gbp",
        "churn_value_at_risk": "churn_gbp_at_risk", "churn_90d": "actually_churned",
        "cancel_90d": "actually_cancelled", "future_revenue_90d": "next_90d_revenue_gbp"}
c = c[list(cols)].rename(columns=cols)
c = c.round({"total_spend_gbp": 2, "avg_order_value_gbp": 2, "units_per_order": 1, "past_cancel_rate": 4,
             "churn_probability": 4, "cancel_probability": 4, "typical_90d_spend_gbp": 2,
             "churn_gbp_at_risk": 2, "next_90d_revenue_gbp": 2})
c.to_csv(OUT + "customers.csv", index=False)

for f in sorted(os.listdir(OUT)):
    print(f"{f:<26} {os.path.getsize(OUT + f) / 1024:7.1f} KB")
