"""
Step 2 - Exploratory analysis and process audit.

Sizes each leak, finds where it concentrates, and checks whether the patterns are
statistically real. Prints every number used later in the memo and resume, saves
supporting tables to outputs/step2/ and charts to outputs/step2/charts/.
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

P = "data/processed/"
OUT = "outputs/step2/"
os.makedirs(OUT + "charts", exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

sales = pd.read_parquet(P + "sales.parquet")
cancels = pd.read_parquet(P + "cancellations.parquet")
writeoffs = pd.read_parquet(P + "stock_writeoffs.parquet")
nonprod = pd.read_parquet(P + "non_product.parquet")

TOTAL = sales["Revenue"].sum()
gbp = lambda x: f"£{x:,.0f}"
pct = lambda x: f"{x:.1%}"

# Chart style: one blue for single series, quiet axes, no top/right spines.
BLUE, ORANGE, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT + "charts/" + name, dpi=160)
    plt.close(fig)

def section(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# =============================================================== A. Leakage sizing
section("A. LEAKAGE SIZING")
# Value write-offs at each product's median selling price (an upper bound on cost).
med_price = sales.groupby("StockCode")["Price"].median()
writeoffs["EstValue"] = -writeoffs["Quantity"] * writeoffs["StockCode"].map(med_price)
manual = nonprod[nonprod["StockCode"] == "M"]
leaks = pd.DataFrame([
    ["Customer cancellations", -cancels["Revenue"].sum(), "lost sales"],
    ["Stock write-offs (at selling price)", writeoffs["EstValue"].sum(), "upper bound"],
    ["Manual credits (gross)", -manual.loc[manual["Revenue"] < 0, "Revenue"].sum(), "unexplained reversals"],
    ["Amazon fees (net)", -nonprod.loc[nonprod["StockCode"] == "AMAZONFEE", "Revenue"].sum(), "channel cost"],
    ["Bad debt, bank, CRUK, discounts", -nonprod.loc[nonprod["StockCode"].isin(["B", "BANK CHARGES", "CRUK", "D"]), "Revenue"].sum(), "finance"],
], columns=["Leak", "Value", "Note"])
leaks["% of sales"] = leaks["Value"] / TOTAL
print(leaks.to_string(index=False, formatters={"Value": gbp, "% of sales": pct}))
print(f"\nProduct sales {gbp(TOTAL)}; sales with no customer ID "
      f"{gbp(sales.loc[~sales.HasCustomer, 'Revenue'].sum())} "
      f"({pct(sales.loc[~sales.HasCustomer, 'Revenue'].sum() / TOTAL)})")
leaks.to_csv(OUT + "leak_sizing.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 3.4))
d = leaks.sort_values("Value")
ax.barh(d["Leak"], d["Value"] / 1e3, color=BLUE, height=0.6)
for y, v in enumerate(d["Value"]):
    ax.text(v / 1e3 + 4, y, f"£{v/1e3:,.0f}k", va="center", color=INK, fontsize=9)
ax.set_xlabel("£ thousands")
ax.set_title("Where value leaks out, Dec 2010 – Dec 2011")
ax.grid(axis="y", visible=False)
save(fig, "01_leak_sizing.png")


# =============================================================== B. Revenue trend
section("B. MONTHLY TREND")
m = (sales.groupby("Month")["Revenue"].sum().to_frame("Sales")
     .join(-cancels.groupby("Month")["Revenue"].sum().rename("Cancelled")))
m["Cancel %"] = m["Cancelled"] / m["Sales"]
m["Customers"] = sales.groupby("Month")["CustomerID"].nunique()
print(m.to_string(formatters={"Sales": gbp, "Cancelled": gbp, "Cancel %": pct}))
full = m.drop(index="2011-12")
print(f"\nPeak Sep-Nov avg {gbp(full.loc['2011-09':'2011-11','Sales'].mean())} vs "
      f"Jan-Aug avg {gbp(full.loc['2011-01':'2011-08','Sales'].mean())}")

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(full.index, full["Sales"] / 1e3, color=BLUE, lw=2, marker="o", ms=4)
ax.set_ylabel("£ thousands")
ax.set_title("Monthly product sales (Dec 2011 excluded: 9 days only)")
ax.tick_params(axis="x", rotation=45)
ax.set_ylim(0)
save(fig, "02_monthly_sales.png")


# =============================================================== C. Cancellations
section("C. CANCELLATIONS – WHERE DO THEY CONCENTRATE?")
cv = -cancels["Revenue"]
print(f"Cancel lines with a customer ID: {pct(cancels.HasCustomer.mean())}")

# C1. Products: Pareto of cancelled £
sku_c = cancels.assign(V=cv).groupby(["StockCode", "Description"])["V"].sum().sort_values(ascending=False)
share = sku_c.cumsum() / sku_c.sum()
n80 = int((share < 0.8).sum()) + 1
print(f"Products with any cancellation: {len(sku_c):,}; {n80:,} of them "
      f"({pct(n80/len(sku_c))}) make up 80% of cancelled £")
print("Top 10 products by cancelled £:")
print(sku_c.head(10).map(gbp).to_string())

# C2. Product cancel RATE (cancelled units / sold units), for products with real volume
sold_u = sales.groupby("StockCode")["Quantity"].sum()
canc_u = -cancels.groupby("StockCode")["Quantity"].sum()
rate = pd.DataFrame({"sold": sold_u, "cancelled": canc_u}).fillna(0)
rate = rate[rate["sold"] >= 500]
rate["rate"] = rate["cancelled"] / rate["sold"]
print(f"\nProduct cancel rate (products with >=500 units sold, n={len(rate):,}): "
      f"median {pct(rate['rate'].median())}, 90th pct {pct(rate['rate'].quantile(.9))}")
rate.sort_values("rate", ascending=False).to_csv(OUT + "product_cancel_rate.csv")

# C3. Customers
cust_sales = sales[sales.HasCustomer].groupby("CustomerID")["Revenue"].sum()
cust_canc = cancels[cancels.HasCustomer].assign(V=cv).groupby("CustomerID")["V"].sum()
cc = pd.DataFrame({"sales": cust_sales, "cancelled": cust_canc}).fillna(0)
cc["ever_cancelled"] = cc["cancelled"] > 0
cc_sorted = cc["cancelled"].sort_values(ascending=False)
top10pct = int(len(cc[cc.ever_cancelled]) * 0.10)
print(f"\nCustomers who ever cancelled: {cc.ever_cancelled.sum():,} of {len(cc):,} ({pct(cc.ever_cancelled.mean())})")
print(f"Top 10% of cancelling customers ({top10pct}) = {pct(cc_sorted.head(top10pct).sum()/cc_sorted.sum())} of customer-cancelled £")
print(f"Median annual spend: cancellers {gbp(cc.loc[cc.ever_cancelled,'sales'].median())} vs "
      f"non-cancellers {gbp(cc.loc[~cc.ever_cancelled,'sales'].median())}")
u = stats.mannwhitneyu(cc.loc[cc.ever_cancelled, "sales"], cc.loc[~cc.ever_cancelled, "sales"])
print(f"  Mann-Whitney U test p = {u.pvalue:.1e}  -> cancellers really are bigger buyers")

# C4. Countries (min £50k sales so rates are meaningful)
ctry = pd.DataFrame({"sales": sales.groupby("Country")["Revenue"].sum(),
                     "cancelled": cancels.assign(V=cv).groupby("Country")["V"].sum()}).fillna(0)
ctry = ctry[ctry["sales"] >= 50_000]
ctry["rate"] = ctry["cancelled"] / ctry["sales"]
print("\nCancel rate by country (>= £50k sales):")
print(ctry.sort_values("rate", ascending=False).to_string(formatters={"sales": gbp, "cancelled": gbp, "rate": pct}))
# Chi-square: is the share of invoices that are cancellations different UK vs rest?
inv = pd.concat([sales[["Invoice", "Country"]].drop_duplicates().assign(c=0),
                 cancels[["Invoice", "Country"]].drop_duplicates().assign(c=1)], ignore_index=True)
inv["UK"] = inv["Country"] == "United Kingdom"
tab = pd.crosstab(inv["UK"], inv["c"])
chi = stats.chi2_contingency(tab)
print(f"Share of invoices that are cancellations: UK {pct(tab.loc[True,1]/tab.loc[True].sum())}, "
      f"non-UK {pct(tab.loc[False,1]/tab.loc[False].sum())}; chi-square p = {chi.pvalue:.1e}")

# C5. Order size: do bigger orders get cancelled more?
fig, ax = plt.subplots(figsize=(8, 3.4))
x = np.arange(1, len(share) + 1) / len(share)
ax.plot(x * 100, share.values * 100, color=BLUE, lw=2)
ax.axhline(80, color=MUTED, lw=0.8, ls="--")
ax.axvline(n80 / len(share) * 100, color=MUTED, lw=0.8, ls="--")
ax.text(n80 / len(share) * 100 + 1, 40, f"{pct(n80/len(share))} of products\n= 80% of cancelled £", color=INK, fontsize=9)
ax.set_xlabel("% of products (ranked by cancelled £)")
ax.set_ylabel("cumulative % of cancelled £")
ax.set_title("Cancellations are concentrated in a few products")
ax.set_xlim(0, 100); ax.set_ylim(0, 101)
save(fig, "03_cancel_pareto.png")


# =============================================================== D. Stock write-offs
section("D. STOCK WRITE-OFFS")
def reason(desc):
    d = str(desc).lower()
    if re.search(r"damag|broken|crush|smash|wet|mould|rotten|cracked", d): return "Damaged"
    if re.search(r"thrown|destroy|unsaleable|dispos|trash|bin", d): return "Destroyed/unsaleable"
    if re.search(r"lost|missing|stolen|found", d): return "Lost/missing"
    if re.search(r"check|count|stock|adjust|\?", d): return "Stock check / unexplained"
    if re.search(r"sold as|dotcom|amazon|ebay|set", d): return "Channel / set re-coding"
    if d in ("nan", "none", ""): return "No reason given"
    return "Other notes"
writeoffs["Reason"] = writeoffs["Description"].map(reason)
r = writeoffs.groupby("Reason").agg(lines=("Invoice", "size"), units=("Quantity", lambda q: -q.sum()),
                                    est_value=("EstValue", "sum")).sort_values("est_value", ascending=False)
r["% value"] = r["est_value"] / r["est_value"].sum()
print(r.to_string(formatters={"est_value": gbp, "% value": pct}))
print(f"Write-offs with no reason or an unexplained note: "
      f"{pct(r.loc[r.index.isin(['No reason given','Stock check / unexplained','Other notes']),'est_value'].sum()/r['est_value'].sum())} of value")
r.to_csv(OUT + "writeoff_reasons.csv")
top_wo = writeoffs.groupby("StockCode")["EstValue"].sum().sort_values(ascending=False)
print(f"Top 20 products = {pct(top_wo.head(20).sum()/top_wo.sum())} of write-off value")


# =============================================================== E. Manual adjustments
section("E. MANUAL ADJUSTMENTS")
mc = manual[manual["Revenue"] < 0]
print(f"Manual credit lines: {len(mc):,}; total {gbp(-mc['Revenue'].sum())}; "
      f"median {gbp(-mc['Revenue'].median())}; largest {gbp(-mc['Revenue'].min())}")
print(f"  with a customer ID: {pct(mc['CustomerID'].notna().mean())}; "
      f"largest 10 lines = {pct(mc['Revenue'].nsmallest(10).sum()/mc['Revenue'].sum())} of manual credit £")
print("  (No reason codes exist for manual credits - every one is just 'Manual'.)")


# =============================================================== F. Missing customer ID
section("F. SALES WITH NO CUSTOMER ID")
f_ = sales.groupby("Month").apply(lambda g: g.loc[~g.HasCustomer, "Revenue"].sum() / g["Revenue"].sum(), include_groups=False)
print("Share of revenue with no customer, by month:")
print(f_.map(pct).to_string())
fc = sales.groupby("Country").apply(lambda g: (~g.HasCustomer).mean(), include_groups=False)
print(f"Countries with any missing IDs: {(fc > 0).sum()} of {len(fc)}; UK line share missing {pct(fc['United Kingdom'])}")
bas = sales.groupby(["Invoice", "HasCustomer"])["Revenue"].sum().reset_index()
print(f"Median basket: with ID {gbp(bas[bas.HasCustomer].Revenue.median())}, without {gbp(bas[~bas.HasCustomer].Revenue.median())}")
print(f"Invoices with no ID: {bas[~bas.HasCustomer].Invoice.nunique():,}")

fig, ax = plt.subplots(figsize=(8, 3.2))
ff = f_.drop(index="2011-12")
ax.bar(ff.index, ff.values * 100, color=BLUE, width=0.6)
ax.set_ylabel("% of monthly revenue")
ax.set_title("Revenue recorded with no customer ID, by month")
ax.tick_params(axis="x", rotation=45)
save(fig, "04_missing_id_by_month.png")


# =============================================================== G. Customers
section("G. CUSTOMER BEHAVIOUR")
k = sales[sales.HasCustomer]
cust = k.groupby("CustomerID").agg(rev=("Revenue", "sum"), orders=("Invoice", "nunique"),
                                   first=("InvoiceDate", "min"), last=("InvoiceDate", "max"))
cs = cust["rev"].sort_values(ascending=False)
print(f"Customers {len(cust):,}; top 20% = {pct(cs.head(int(len(cs)*.2)).sum()/cs.sum())} of revenue; "
      f"top 1% = {pct(cs.head(int(len(cs)*.01)).sum()/cs.sum())}")
print(f"One-time buyers: {pct((cust.orders == 1).mean())}; median orders {cust.orders.median():.0f}")
# Gap between consecutive orders, per customer
o = k.groupby(["CustomerID", "Invoice"])["InvoiceDate"].min().reset_index().sort_values(["CustomerID", "InvoiceDate"])
o["gap"] = o.groupby("CustomerID")["InvoiceDate"].diff().dt.days
g = o["gap"].dropna()
g = g[g > 0]
print(f"Days between repeat orders: median {g.median():.0f}, 75th pct {g.quantile(.75):.0f}, 90th pct {g.quantile(.9):.0f}")

# Monthly cohort retention: of customers who first bought in month X, % buying again in month X+n
# December 2011 is excluded: 9 days of data would make every cohort look like it collapsed.
# Note the Dec 2010 "cohort" is really everyone already a customer when the data starts.
kk = k[~k["PartialMonth"]]
k2 = kk.assign(cm=kk.groupby("CustomerID")["InvoiceDate"].transform("min").dt.to_period("M"),
               om=kk["InvoiceDate"].dt.to_period("M"))
k2["n"] = (k2["om"] - k2["cm"]).apply(lambda x: x.n)
coh = k2.groupby(["cm", "n"])["CustomerID"].nunique().unstack()
ret = coh.div(coh[0], axis=0)
print("Cohort retention (share of a cohort active n months later), first 4 cohorts:")
print(ret.iloc[:4, :7].map(lambda v: "" if pd.isna(v) else f"{v:.0%}").to_string())
new = ret.iloc[1:]  # true new-customer cohorts only
print(f"New-customer cohorts (Jan-Nov 2011): avg month-1 retention {pct(new[1].mean())}, "
      f"month-3 {pct(new[3].mean())}; Dec-2010 base customers month-1 {pct(ret.iloc[0][1])}")
ret.to_csv(OUT + "cohort_retention.csv")

fig, ax = plt.subplots(figsize=(8, 4.2))
show = ret.iloc[:-1, 1:12] * 100
im = ax.imshow(show.values, cmap="Blues", vmin=0, vmax=50, aspect="auto")
ax.set_xticks(range(show.shape[1]), [str(c) for c in show.columns])
ax.set_yticks(range(show.shape[0]), [str(i) for i in show.index])
ax.set_xlabel("months since first purchase"); ax.set_ylabel("first-purchase month")
ax.set_title("Cohort retention: % of each cohort buying again")
ax.grid(False)
for i in range(show.shape[0]):
    for j in range(show.shape[1]):
        v = show.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 30 else INK)
fig.colorbar(im, ax=ax, label="% retained", shrink=0.8)
save(fig, "05_cohort_retention.png")

cc.to_csv(OUT + "customer_sales_vs_cancels.csv")
print("\nSaved tables to", OUT, "and charts to", OUT + "charts/")
