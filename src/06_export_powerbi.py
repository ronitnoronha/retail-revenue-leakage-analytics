"""
Step 6 - Export a star schema for the Power BI dashboard.

Facts (one row per event)          Dimensions (one row per thing)
  fact_sales.csv                      dim_date.csv
  fact_cancellations.csv              dim_product.csv
  fact_writeoffs.csv                  dim_customer.csv
  fact_adjustments.csv
Reference tables (small, pre-computed in Python)
  ref_leak_summary.csv, ref_model_churn.csv, ref_model_cancel.csv,
  ref_retention_targeting.csv, ref_campaign_sensitivity.csv, ref_cancel_policy.csv

Why a star schema: Power BI filters flow from dimensions to facts. Keeping dates, products and
customers in their own tables means one slicer (e.g. Country) filters every fact table at once.
"""
import os
import re
import pandas as pd

P, S2, S4, S5, OUT = "data/processed/", "outputs/step2/", "outputs/step4/", "outputs/step5/", "powerbi/data/"
os.makedirs(OUT, exist_ok=True)

sales = pd.read_parquet(P + "sales.parquet")
cancels = pd.read_parquet(P + "cancellations.parquet")
writeoffs = pd.read_parquet(P + "stock_writeoffs.parquet")
nonprod = pd.read_parquet(P + "non_product.parquet")
feat = pd.read_parquet(P + "features.parquet")
actions = pd.read_csv(S5 + "customer_action_list.csv", dtype={"CustomerID": str})

def date_cols(df):
    df = df.copy()
    df["Date"] = df["InvoiceDate"].dt.date
    return df

# ---------------------------------------------------------------- facts
fact_sales = date_cols(sales)[["Invoice", "Date", "InvoiceDate", "StockCode", "CustomerID", "Country",
                               "Quantity", "Price", "Revenue", "HasCustomer"]]
fact_cancel = date_cols(cancels).assign(CancelValue=lambda d: -d["Revenue"])[
    ["Invoice", "Date", "StockCode", "CustomerID", "Country", "Quantity", "Price", "CancelValue"]]

med_price = sales.groupby("StockCode")["Price"].median()
def reason(desc):
    d = str(desc).lower()
    if re.search(r"damag|broken|crush|smash|wet|mould|rotten|cracked", d): return "Damaged"
    if re.search(r"thrown|destroy|unsaleable|dispos|trash|bin", d): return "Destroyed/unsaleable"
    if re.search(r"lost|missing|stolen|found", d): return "Lost/missing"
    if re.search(r"check|count|stock|adjust|\?", d): return "Stock check / unexplained"
    if re.search(r"sold as|dotcom|amazon|ebay|set", d): return "Channel / set re-coding"
    if d in ("nan", "none", ""): return "No reason given"
    return "Other notes"
fact_wo = date_cols(writeoffs).assign(
    Units=lambda d: -d["Quantity"],
    EstValue=lambda d: -d["Quantity"] * d["StockCode"].map(med_price),
    Reason=lambda d: d["Description"].map(reason))
fact_wo["ReasonRecorded"] = ~fact_wo["Reason"].isin(["No reason given", "Stock check / unexplained", "Other notes"])
fact_wo = fact_wo[["Invoice", "Date", "StockCode", "Units", "EstValue", "Reason", "ReasonRecorded"]]

TYPE = {"POST": "Postage", "DOT": "Postage", "C2": "Postage", "M": "Manual adjustment", "AMAZONFEE": "Amazon fee",
        "BANK CHARGES": "Bank charges", "D": "Discount", "CRUK": "Charity commission", "B": "Bad debt",
        "S": "Samples", "PADS": "Other"}
fact_adj = date_cols(nonprod).assign(
    Type=lambda d: d["StockCode"].map(TYPE).fillna("Gift voucher"),
    IsCredit=lambda d: d["Revenue"] < 0)
fact_adj = fact_adj[["Invoice", "Date", "StockCode", "Type", "CustomerID", "Country", "Revenue", "IsCredit"]]

# ---------------------------------------------------------------- dimensions
dates = pd.date_range("2010-12-01", "2011-12-31", freq="D")
dim_date = pd.DataFrame({"Date": dates.date, "Year": dates.year, "MonthNum": dates.month,
                         "Month": dates.strftime("%b %Y"), "MonthStart": dates.to_period("M").start_time.date,
                         "Weekday": dates.strftime("%a"), "WeekdayNum": dates.weekday + 1,
                         "Quarter": "Q" + dates.quarter.astype(str),
                         "IsPeakSeason": dates.month.isin([9, 10, 11]),
                         "IsPartialMonth": (dates.year == 2011) & (dates.month == 12)})

prod = pd.DataFrame({
    "Description": sales.groupby("StockCode")["Description"].first(),
    "SalesGBP": sales.groupby("StockCode")["Revenue"].sum(),
    "UnitsSold": sales.groupby("StockCode")["Quantity"].sum(),
    "MedianPrice": med_price,
    "CancelGBP": fact_cancel.groupby("StockCode")["CancelValue"].sum(),
    "WriteOffGBP": fact_wo.groupby("StockCode")["EstValue"].sum(),
}).fillna(0)
prod["CancelRate"] = (prod["CancelGBP"] / prod["SalesGBP"]).where(prod["SalesGBP"] > 0)
prod["ReviewFlag"] = (prod["CancelRate"] >= 0.05) & (prod["SalesGBP"] >= 1000)
# Codes seen only in cancellations / write-offs still need a row so relationships don't break.
extra = sorted((set(fact_cancel.StockCode) | set(fact_wo.StockCode)) - set(prod.index))
prod = prod.reindex(prod.index.union(extra))
prod["Description"] = prod["Description"].fillna("(not sold in period)")
dim_product = prod.reset_index().rename(columns={"index": "StockCode"})

ks = sales[sales.HasCustomer]
cust = ks.groupby("CustomerID").agg(Country=("Country", lambda c: c.mode().iloc[0]),
                                    FirstPurchase=("InvoiceDate", "min"), LastPurchase=("InvoiceDate", "max"),
                                    Orders=("Invoice", "nunique"), SalesGBP=("Revenue", "sum"))
cust["CancelGBP"] = fact_cancel.groupby("CustomerID")["CancelValue"].sum()
cust["CancelGBP"] = cust["CancelGBP"].fillna(0)
cust["FirstPurchaseMonth"] = cust["FirstPurchase"].dt.to_period("M").astype(str)
cust["OneTimeBuyer"] = cust["Orders"] == 1
# Revenue tier: top 20% of customers by spend vs the rest
cust["ValueTier"] = pd.qcut(cust["SalesGBP"].rank(method="first"), [0, .5, .8, 1],
                            labels=["Bottom 50%", "Middle 30%", "Top 20%"]).astype(str)
act = actions.set_index("CustomerID")[["action", "p_churn", "p_cancel", "value_90d", "churn_value_at_risk",
                                       "cancel_exp_loss", "clv_90d", "churn_90d", "cancel_90d"]]
act.columns = ["Action", "ChurnProb", "CancelProb", "Typical90dSpend", "ChurnGBPAtRisk",
               "CancelExpLossGBP", "CLV90d", "ActualChurned", "ActualCancelled"]
dim_customer = cust.join(act, how="left").reset_index()
dim_customer["Action"] = dim_customer["Action"].fillna("Not scored (inactive >180 days on 10 Sep)")
# Anonymous sales get a placeholder customer so they can still be counted in the model.
anon = pd.DataFrame([{"CustomerID": "UNKNOWN", "Country": "Unknown", "Action": "No customer ID",
                      "ValueTier": "Unknown"}])
for f in (fact_sales, fact_cancel, fact_adj):
    f["CustomerID"] = f["CustomerID"].fillna("UNKNOWN")
# Customers seen only in cancellations/adjustments (they bought before Dec 2010) still need a row.
seen = set(fact_cancel.CustomerID) | set(fact_adj.CustomerID)
only = sorted(seen - set(dim_customer.CustomerID) - {"UNKNOWN"})
pre = pd.DataFrame({"CustomerID": only, "Action": "No purchases in period", "ValueTier": "No purchases",
                    "Orders": 0, "SalesGBP": 0.0})
pre["Country"] = pre["CustomerID"].map(pd.concat([fact_cancel, fact_adj]).groupby("CustomerID")["Country"].first())
pre["CancelGBP"] = pre["CustomerID"].map(fact_cancel.groupby("CustomerID")["CancelValue"].sum()).fillna(0)
dim_customer = pd.concat([dim_customer, pre, anon], ignore_index=True)
print(f"Added {len(only)} customers with no purchases in the period (cancellations/adjustments only)")

# ---------------------------------------------------------------- reference tables
pd.read_csv(S2 + "leak_sizing.csv").to_csv(OUT + "ref_leak_summary.csv", index=False)
pd.read_csv(S4 + "churn_model_comparison.csv").to_csv(OUT + "ref_model_churn.csv", index=False)
pd.read_csv(S4 + "cancel_model_comparison.csv").to_csv(OUT + "ref_model_cancel.csv", index=False)
pd.read_csv(S5 + "retention_targeting.csv").to_csv(OUT + "ref_retention_targeting.csv", index=False)
pd.read_csv(S5 + "retention_campaign_sensitivity.csv").to_csv(OUT + "ref_campaign_sensitivity.csv", index=False)
pd.read_csv(S5 + "cancel_policy_targeting.csv").to_csv(OUT + "ref_cancel_policy.csv", index=False)

# ---------------------------------------------------------------- write + integrity checks
tables = {"fact_sales": fact_sales, "fact_cancellations": fact_cancel, "fact_writeoffs": fact_wo,
          "fact_adjustments": fact_adj, "dim_date": dim_date, "dim_product": dim_product,
          "dim_customer": dim_customer}
for name, t in tables.items():
    t.to_csv(OUT + name + ".csv", index=False)
    print(f"{name:<20} {len(t):>8,} rows  {os.path.getsize(OUT + name + '.csv') / 1e6:6.1f} MB")

# Every key in a fact table must exist in its dimension, or Power BI shows "(Blank)".
for f in ("fact_sales", "fact_cancellations", "fact_writeoffs"):
    assert set(tables[f]["StockCode"]) <= set(dim_product["StockCode"]), f + " product key missing"
for f in ("fact_sales", "fact_cancellations", "fact_adjustments"):
    missing = set(tables[f]["CustomerID"]) - set(dim_customer["CustomerID"])
    assert not missing, f"{f}: {len(missing)} customer keys missing"
assert dim_customer["CustomerID"].is_unique and dim_product["StockCode"].is_unique
# Totals reconcile with the earlier steps
print(f"\nCheck: sales £{fact_sales.Revenue.sum():,.0f} | cancels £{fact_cancel.CancelValue.sum():,.0f} | "
      f"write-offs £{fact_wo.EstValue.sum():,.0f} | customers scored {dim_customer.ChurnProb.notna().sum():,}")
