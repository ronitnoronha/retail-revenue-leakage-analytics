"""
Build the small, pre-aggregated CSVs the online dashboard reads.
Run from the project root after Steps 1-5:  python dashboard/build_data.py
Output: dashboard/data/*.csv
"""
import os
import re
import numpy as np
import pandas as pd

P, S2, S4, S5, OUT = "data/processed/", "outputs/step2/", "outputs/step4/", "outputs/step5/", "dashboard/data/"
os.makedirs(OUT, exist_ok=True)

raw = pd.read_csv("data/raw/online_retail_II_2010_2011.csv", encoding="latin-1", dtype={"Invoice": str, "StockCode": str})
sales = pd.read_parquet(P + "sales.parquet")
cancels = pd.read_parquet(P + "cancellations.parquet")
writeoffs = pd.read_parquet(P + "stock_writeoffs.parquet")
nonprod = pd.read_parquet(P + "non_product.parquet")
TOTAL = sales["Revenue"].sum()

# ---------------------------------------------------------------- overview (one row)
raw_rev = raw["Quantity"] * raw["Price"]
raw_c = raw["Invoice"].str.startswith("C")
raw_sales = raw_rev[~raw_c & (raw["Quantity"] > 0) & (raw["Price"] > 0)].sum()
pd.DataFrame([{
    "raw_transactions": len(raw),
    "product_sales_gbp": round(TOTAL),
    "invoices": sales["Invoice"].nunique(),
    "customers": sales["CustomerID"].nunique(),
    "countries": sales["Country"].nunique(),
    "raw_cancel_pct": round(-raw_rev[raw_c].sum() / raw_sales, 4),
    "clean_cancel_pct": round(-cancels["Revenue"].sum() / TOTAL, 4),
    "unattributed_gbp": round(sales.loc[~sales.HasCustomer, "Revenue"].sum()),
    "unattributed_pct": round(sales.loc[~sales.HasCustomer, "Revenue"].sum() / TOTAL, 4),
}]).to_csv(OUT + "overview.csv", index=False)

# ---------------------------------------------------------------- leaks
cause = {
    "Stock write-offs (at selling price)": ("Process", "Half the value has no recorded reason"),
    "Customer cancellations": ("Customer", "10% of products cause 50% of product cancel £"),
    "Amazon fees (net)": ("Channel cost", "A margin question for the channel, not an error"),
    "Manual credits (gross)": ("Process", "No reason codes; 10 credits are 52% of the value"),
    "Bad debt, bank, CRUK, discounts": ("Finance", "Bad debt, bank charges, commission, discounts"),
}
lk = pd.read_csv(S2 + "leak_sizing.csv")
lk = pd.DataFrame({"leak": lk["Leak"], "value_gbp": lk["Value"].round(), "pct_of_sales": lk["% of sales"].round(4),
                   "kind": lk["Leak"].map(lambda x: cause[x][0]), "root_cause": lk["Leak"].map(lambda x: cause[x][1])})
lk.sort_values("value_gbp", ascending=False).to_csv(OUT + "leaks.csv", index=False)

# ---------------------------------------------------------------- monthly
med_price = sales.groupby("StockCode")["Price"].median()
writeoffs["EstValue"] = -writeoffs["Quantity"] * writeoffs["StockCode"].map(med_price)
writeoffs["Month"] = writeoffs["InvoiceDate"].dt.to_period("M").astype(str)
nonprod["Month"] = nonprod["InvoiceDate"].dt.to_period("M").astype(str)
mc = nonprod[(nonprod.StockCode == "M") & (nonprod.Revenue < 0)]
m = pd.DataFrame({
    "sales_gbp": sales.groupby("Month")["Revenue"].sum(),
    "cancelled_gbp": -cancels.groupby("Month")["Revenue"].sum(),
    "unattributed_gbp": sales[~sales.HasCustomer].groupby("Month")["Revenue"].sum(),
    "manual_credits_gbp": -mc.groupby("Month")["Revenue"].sum(),
    "writeoff_gbp": writeoffs.groupby("Month")["EstValue"].sum(),
    "active_customers": sales.groupby("Month")["CustomerID"].nunique(),
}).fillna(0)
m["cancel_rate"] = m["cancelled_gbp"] / m["sales_gbp"]
m["unattributed_pct"] = m["unattributed_gbp"] / m["sales_gbp"]
m["partial_month"] = m.index == "2011-12"
m = m.round({"sales_gbp": 0, "cancelled_gbp": 0, "unattributed_gbp": 0, "manual_credits_gbp": 0,
             "writeoff_gbp": 0, "cancel_rate": 4, "unattributed_pct": 4})
m.rename_axis("month").reset_index().to_csv(OUT + "monthly.csv", index=False)

# ---------------------------------------------------------------- write-off reasons
wr = pd.read_csv(S2 + "writeoff_reasons.csv")
wr = pd.DataFrame({"reason": wr["Reason"], "lines": wr["lines"], "units": wr["units"],
                   "value_gbp": wr["est_value"].round(), "share": wr["% value"].round(4),
                   "reason_recorded": ~wr["Reason"].isin(["No reason given", "Stock check / unexplained", "Other notes"])})
wr.to_csv(OUT + "writeoff_reasons.csv", index=False)

# ---------------------------------------------------------------- manual credit thresholds
amt = -mc["Revenue"]
pd.DataFrame([{"threshold_gbp": t, "credits_needing_approval": int((amt >= t).sum()), "total_credits": len(amt),
               "pct_of_credits": round((amt >= t).mean(), 4), "value_covered_gbp": round(amt[amt >= t].sum()),
               "pct_of_value": round(amt[amt >= t].sum() / amt.sum(), 4)}
              for t in (100, 250, 500, 1000, 2500)]).to_csv(OUT + "manual_thresholds.csv", index=False)

# ---------------------------------------------------------------- products for review (top 15)
pr = pd.read_csv(S5 + "products_for_review.csv", dtype={"StockCode": str})
pr = pr.rename(columns={"StockCode": "stock_code", "desc": "description", "sold_gbp": "sold_gbp",
                        "cancel_gbp": "cancelled_gbp", "rate": "cancel_rate"})
pr["sold_gbp"] = pr["sold_gbp"].round(); pr["cancelled_gbp"] = pr["cancelled_gbp"].round(); pr["cancel_rate"] = pr["cancel_rate"].round(4)
pr.head(15)[["stock_code", "description", "sold_gbp", "cancelled_gbp", "cancel_rate"]].to_csv(OUT + "products_review.csv", index=False)

# ---------------------------------------------------------------- customer actions
al = pd.read_csv(S5 + "customer_action_list.csv", dtype={"CustomerID": str})
ag = al.groupby("action").agg(customers=("CustomerID", "size"), churn_risk_gbp=("churn_value_at_risk", "sum"),
                              cancel_risk_gbp=("cancel_exp_loss", "sum"), actual_churn_rate=("churn_90d", "mean"),
                              actual_cancel_rate=("cancel_90d", "mean")).reset_index()
ag["action_label"] = ag["action"].str.replace(r"^\d\. ", "", regex=True)
ag = ag.round({"churn_risk_gbp": 0, "cancel_risk_gbp": 0, "actual_churn_rate": 4, "actual_cancel_rate": 4})
ag[["action", "action_label", "customers", "churn_risk_gbp", "cancel_risk_gbp", "actual_churn_rate",
    "actual_cancel_rate"]].to_csv(OUT + "action_groups.csv", index=False)

feat = pd.read_parquet(P + "features.parquet").query("split == 'test'")
ctry = sales[sales.HasCustomer].groupby("CustomerID")["Country"].agg(lambda c: c.mode().iloc[0])
al = al.merge(feat[["CustomerID", "recency_days", "frequency"]], on="CustomerID", how="left")
top = al.sort_values("churn_value_at_risk", ascending=False).head(25)
pd.DataFrame({"customer_id": top["CustomerID"], "country": top["CustomerID"].map(ctry),
              "orders_before_sep": top["frequency"], "days_since_last_order": top["recency_days"],
              "churn_probability": top["p_churn"].round(3), "typical_90d_spend_gbp": top["value_90d"].round(),
              "gbp_at_risk": top["churn_value_at_risk"].round(), "actually_churned": top["churn_90d"].map({1: "Yes", 0: "No"})
              }).to_csv(OUT + "retention_targets.csv", index=False)

# ---------------------------------------------------------------- retention gains curve (test set)
lost = np.where(al["churn_90d"] == 1, al["value_90d"], 0.0)
def curve(score):
    order = np.argsort(-score.values)
    cum = np.concatenate([[0], np.cumsum(lost[order])]) / lost.sum()
    idx = np.round(np.linspace(0, len(al), 101)).astype(int)
    return cum[idx]
g = pd.DataFrame({"pct_contacted": np.arange(0, 101) / 100,
                  "model_expected_gbp": curve(al["churn_value_at_risk"]),
                  "recency_rule": curve(al["recency_days"]),
                  "biggest_spenders": curve(al["value_90d"]),
                  "random_expected": np.arange(0, 101) / 100}).round(4)
g.to_csv(OUT + "retention_gains.csv", index=False)

# ---------------------------------------------------------------- campaign sensitivity
cs = pd.read_csv(S5 + "retention_campaign_sensitivity.csv")
pd.DataFrame({"save_rate": cs["Save rate"], "cost_per_contact_gbp": cs["Cost / contact"],
              "net_gbp_model": cs["Net £ (model)"].round(), "net_gbp_recency_rule": cs["Net £ (recency rule)"].round()}
             ).to_csv(OUT + "campaign.csv", index=False)

# ---------------------------------------------------------------- models
FAM = {"Logistic Regression": "Linear", "Cox Proportional Hazards": "Survival", "LightGBM": "Gradient boosting",
       "Random Survival Forest": "Survival + ensemble", "Random Forest": "Tree ensemble", "MLP Neural Net": "Neural network",
       "BG/NBD": "Probabilistic", "Baseline: recency rule": "Rule", "Baseline: spend rule": "Rule"}
for src, dst in (("churn_model_comparison.csv", "churn_models.csv"), ("cancel_model_comparison.csv", "cancel_models.csv")):
    t = pd.read_csv(S4 + src)
    out = pd.DataFrame({"model": t["Model"], "family": t["Model"].map(FAM),
                        "cv_roc_auc": t["CV ROC-AUC"].round(3), "cv_sd": t["CV ± sd"].round(3),
                        "test_roc_auc": t["Test ROC-AUC"].round(3), "test_pr_auc": t["Test PR-AUC"].round(3),
                        "brier": t["Test Brier"].round(3), "top20_capture": t["Top-20% capture"].round(4),
                        "fit_seconds": t["Fit time (s)"].round(1)})
    if "Top-20% £ capture" in t:
        out["top20_gbp_capture"] = t["Top-20% £ capture"].round(4)
    out.to_csv(OUT + dst, index=False)

cp = pd.read_csv(S5 + "cancel_policy_targeting.csv")
cp = cp[cp["Flag"].str.startswith("top 5%")]
pd.DataFrame({"strategy": cp["Strategy"], "pct_cancelled_gbp_covered": cp["% of cancelled £"].round(4),
              "pct_revenue_touched": cp["% of next-90d revenue touched"].round(4)}).to_csv(OUT + "cancel_policy.csv", index=False)

for f in sorted(os.listdir(OUT)):
    print(f"{f:<26} {len(pd.read_csv(OUT + f)):>4} rows")
print("\nmodel reach at 10%:", g.loc[10, "model_expected_gbp"], "recency:", g.loc[10, "recency_rule"])
