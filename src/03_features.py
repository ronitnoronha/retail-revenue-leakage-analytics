"""
Step 3 - Feature engineering with time-based snapshots.

Turns ~520k transaction lines into one row per (customer, snapshot date).
For every snapshot ("cutoff") date:
  * features use ONLY data strictly before the cutoff  (the past)
  * labels use ONLY the 90 days after the cutoff         (the future we predict)
This is what stops data leakage: the model never sees anything it couldn't
have known on the day the prediction would be made.

Snapshots
  train : 2011-03-01, 2011-04-01, 2011-05-01, 2011-06-01  (label windows end by 2011-08-30)
  test  : 2011-09-10                                        (label window = last 90 days of data)
The test label window starts after every training label window has ended, so the
test set is a genuine "future" the models were never trained on.

Output: data/processed/features.parquet
"""
import numpy as np
import pandas as pd

P = "data/processed/"
HORIZON = pd.Timedelta(days=90)     # how far ahead we predict
ACTIVE = pd.Timedelta(days=180)     # who counts as a current customer at the cutoff
TRAIN_CUTOFFS = ["2011-03-01", "2011-04-01", "2011-05-01", "2011-06-01"]
TEST_CUTOFF = "2011-09-10"
WHOLESALE_QTY = 24                  # a line of 24+ units is a trade-size line

sales = pd.read_parquet(P + "sales.parquet")
cancels = pd.read_parquet(P + "cancellations.parquet")
nonprod = pd.read_parquet(P + "non_product.parquet")

# Customer-level modelling needs a customer: drop anonymous lines here (they stay in Step 2 totals).
sales = sales[sales["HasCustomer"]].copy()
cancels = cancels[cancels["HasCustomer"]].copy()
cancels["CancelValue"] = -cancels["Revenue"]
manual = nonprod[(nonprod["StockCode"] == "M") & nonprod["CustomerID"].notna() & (nonprod["Revenue"] < 0)]

# One row per order makes recency / frequency / gap features simple.
orders = (sales.groupby(["CustomerID", "Invoice"])
          .agg(date=("InvoiceDate", "min"), value=("Revenue", "sum"), lines=("StockCode", "size"),
               units=("Quantity", "sum"), country=("Country", "first"))
          .reset_index())


def build(cutoff: str) -> pd.DataFrame:
    t = pd.Timestamp(cutoff)

    # ---------- split the world at the cutoff
    s_past = sales[sales["InvoiceDate"] < t]
    o_past = orders[orders["date"] < t]
    c_past = cancels[cancels["InvoiceDate"] < t]
    o_future = orders[(orders["date"] >= t) & (orders["date"] < t + HORIZON)]
    c_future = cancels[(cancels["InvoiceDate"] >= t) & (cancels["InvoiceDate"] < t + HORIZON)]

    # ---------- population: customers who bought in the 180 days before the cutoff
    last_buy = o_past.groupby("CustomerID")["date"].max()
    pop = last_buy[last_buy >= t - ACTIVE].index
    o_past = o_past[o_past["CustomerID"].isin(pop)]
    s_past = s_past[s_past["CustomerID"].isin(pop)]

    f = pd.DataFrame(index=pd.Index(pop, name="CustomerID"))

    # ---------- RFM: the classic customer-value trio
    g = o_past.groupby("CustomerID")
    f["recency_days"] = (t - g["date"].max()).dt.days
    f["frequency"] = g.size()
    f["monetary"] = g["value"].sum()
    f["avg_order_value"] = g["value"].mean()
    f["tenure_days"] = (t - g["date"].min()).dt.days

    # ---------- recent activity windows and trend
    for d in (30, 90):
        w = o_past[o_past["date"] >= t - pd.Timedelta(days=d)].groupby("CustomerID")
        f[f"orders_{d}d"] = w.size()
        f[f"revenue_{d}d"] = w["value"].sum()
    prev = o_past[(o_past["date"] >= t - pd.Timedelta(days=180)) & (o_past["date"] < t - pd.Timedelta(days=90))]
    f["revenue_prev_90d"] = prev.groupby("CustomerID")["value"].sum()
    # >1 means spending is rising, <1 falling. +1 in both parts avoids divide-by-zero.
    f[["orders_30d", "orders_90d", "revenue_30d", "revenue_90d", "revenue_prev_90d"]] = \
        f[["orders_30d", "orders_90d", "revenue_30d", "revenue_90d", "revenue_prev_90d"]].fillna(0)
    f["revenue_trend"] = (f["revenue_90d"] + 1) / (f["revenue_prev_90d"] + 1)

    # ---------- purchase rhythm: is this customer "overdue"?
    o_sorted = o_past.sort_values(["CustomerID", "date"])
    gaps = o_sorted.groupby("CustomerID")["date"].diff().dt.days
    gg = gaps.groupby(o_sorted["CustomerID"])
    f["mean_gap_days"] = gg.mean()
    f["std_gap_days"] = gg.std()
    # Single-order customers have no gap: fill with tenure (their only known interval) and 0 spread.
    f["mean_gap_days"] = f["mean_gap_days"].fillna(f["tenure_days"])
    f["std_gap_days"] = f["std_gap_days"].fillna(0)
    f["overdue_ratio"] = f["recency_days"] / (f["mean_gap_days"] + 1)

    # ---------- basket shape: retail shopper vs wholesale buyer
    f["avg_lines_per_order"] = g["lines"].mean()
    f["avg_units_per_order"] = g["units"].mean()
    sp = s_past.groupby("CustomerID")
    f["n_products"] = sp["StockCode"].nunique()
    f["avg_unit_price"] = sp["Price"].mean()
    f["wholesale_line_share"] = sp["Quantity"].apply(lambda q: (q >= WHOLESALE_QTY).mean())

    # ---------- cancellation history (past only)
    cp = c_past[c_past["CustomerID"].isin(pop)].groupby("CustomerID")
    f["n_cancel_invoices"] = cp["Invoice"].nunique()
    f["cancel_value"] = cp["CancelValue"].sum()
    f["days_since_cancel"] = (t - cp["InvoiceDate"].max()).dt.days
    f[["n_cancel_invoices", "cancel_value"]] = f[["n_cancel_invoices", "cancel_value"]].fillna(0)
    f["cancel_rate"] = f["cancel_value"] / f["monetary"]
    # Never cancelled -> treat as "longer ago than any real value" rather than 0 (which would mean "today").
    f["days_since_cancel"] = f["days_since_cancel"].fillna(999)

    # ---------- product risk: share of spend on products that were often cancelled
    # Product cancel rates are recomputed from PAST data only at each cutoff (using all-time
    # rates here would leak future returns into the features).
    su = sales[sales["InvoiceDate"] < t].groupby("StockCode")["Quantity"].sum()
    cu = -cancels[cancels["InvoiceDate"] < t].groupby("StockCode")["Quantity"].sum()
    prate = (cu.reindex(su.index).fillna(0) / su).where(su >= 100, 0)
    risky = prate[prate >= 0.05].index
    spend_risky = s_past[s_past["StockCode"].isin(risky)].groupby("CustomerID")["Revenue"].sum()
    f["risky_product_share"] = (spend_risky.reindex(f.index).fillna(0) / f["monetary"]).clip(0, 1)

    # ---------- manual credits received before the cutoff
    f["n_manual_credits"] = manual[manual["InvoiceDate"] < t].groupby("CustomerID").size().reindex(f.index).fillna(0)

    # ---------- country (5 groups; small countries pooled into "Other")
    ctry = g["country"].agg(lambda c: c.mode().iloc[0])
    for c in ["United Kingdom", "Germany", "France", "EIRE"]:
        f["country_" + {"United Kingdom": "uk"}.get(c, c.lower())] = (ctry == c).astype(int)

    # ---------- labels (future only)
    f["churn_90d"] = (~f.index.isin(o_future["CustomerID"])).astype(int)
    f["cancel_90d"] = f.index.isin(c_future["CustomerID"]).astype(int)
    f["future_revenue_90d"] = o_future.groupby("CustomerID")["value"].sum().reindex(f.index).fillna(0)
    f["future_cancel_value_90d"] = c_future.groupby("CustomerID")["CancelValue"].sum().reindex(f.index).fillna(0)

    f["cutoff"] = t
    f["split"] = "test" if cutoff == TEST_CUTOFF else "train"

    # ---------- leakage guard: nothing in the features may come from on/after the cutoff
    assert s_past["InvoiceDate"].max() < t and c_past["InvoiceDate"].max() < t
    return f.reset_index()


frames = [build(c) for c in TRAIN_CUTOFFS + [TEST_CUTOFF]]
feat = pd.concat(frames, ignore_index=True)

# The last training label window must end before the test window starts.
assert pd.Timestamp(TRAIN_CUTOFFS[-1]) + HORIZON <= pd.Timestamp(TEST_CUTOFF)

feat.to_parquet(P + "features.parquet", index=False)

FEATURES = [c for c in feat.columns if c not in
            ("CustomerID", "cutoff", "split", "churn_90d", "cancel_90d",
             "future_revenue_90d", "future_cancel_value_90d")]
print(f"{len(FEATURES)} features:", ", ".join(FEATURES))
print("\nRows and label rates per snapshot")
print(feat.groupby(["split", "cutoff"]).agg(customers=("CustomerID", "size"),
                                            churn_rate=("churn_90d", "mean"),
                                            cancel_rate=("cancel_90d", "mean")).round(3).to_string())
print(f"\nTrain rows {int((feat.split=='train').sum()):,} "
      f"(unique customers {feat.loc[feat.split=='train','CustomerID'].nunique():,}); "
      f"test rows {int((feat.split=='test').sum()):,}")
print(f"Missing values left: {int(feat[FEATURES].isna().sum().sum())}")

# Quick sanity check: how each feature relates to each label on the training set.
tr = feat[feat.split == "train"]
corr = pd.DataFrame({
    "churn_90d": tr[FEATURES].corrwith(tr["churn_90d"], method="spearman"),
    "cancel_90d": tr[FEATURES].corrwith(tr["cancel_90d"], method="spearman"),
}).round(2)
print("\nSpearman correlation with each label (train)")
print(corr.sort_values("churn_90d").to_string())
