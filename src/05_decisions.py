"""
Step 5 - From model scores to £ decisions.

A. Retention targeting   - who should get a retention contact, and is it worth it?
B. Cancellation policy   - which accounts get an order-confirmation step?
C. Process controls      - rules for write-offs, manual credits, missing customer IDs, risky products.
D. Customer action list  - one row per customer with a recommended action (feeds the dashboard).

Everything is evaluated on the out-of-time test snapshot (scored 10 Sep 2011, outcomes to 9 Dec 2011),
so the "captured £" figures are what would actually have happened.
Assumptions that the data cannot tell us (campaign save rate, contact cost, prevention rate)
are stated explicitly and shown as sensitivity ranges, never as single facts.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P, S2, S4, OUT = "data/processed/", "outputs/step2/", "outputs/step4/", "outputs/step5/"
os.makedirs(OUT + "charts", exist_ok=True)
pd.set_option("display.width", 200)
gbp = lambda x: f"£{x:,.0f}"
pct = lambda x: f"{x:.1%}"
rng = np.random.default_rng(42)

d = pd.read_parquet(S4 + "test_predictions.parquet")
# Bring back the test-snapshot features (recency, past cancellations, ...) for the rule baselines.
tf = pd.read_parquet(P + "features.parquet").query("split == 'test'")
d = d.merge(tf[["CustomerID", "recency_days", "cancel_value", "tenure_days"]], on="CustomerID", how="left")
# A customer's value = their TYPICAL 90-day spend over their whole history (not just the last
# 90 days, which is £0 for anyone already silent and would make silent customers look worthless).
typical90 = lambda df: df["monetary"] * 90 / np.maximum(df["tenure_days"], 90)
d["value_90d"] = typical90(d)
d["p_churn"] = d["churn|Logistic Regression"]       # best churn model (Step 4)
d["p_cancel"] = d["cancel|Logistic Regression"]     # best cancellation model (Step 4)
N = len(d)

def section(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# =============================================================== A. Retention targeting
section("A. RETENTION TARGETING")
# What a churner costs: the revenue they'd have brought if they stayed = their typical 90-day spend.
d["churn_value_at_risk"] = d["p_churn"] * d["value_90d"]         # expected £ lost
lost = np.where(d["churn_90d"] == 1, d["value_90d"], 0)          # £ actually lost (realised)
print(f"Actual churners {int(d.churn_90d.sum()):,} of {N:,}; their typical 90-day spend "
      f"(revenue the business lost) = {gbp(lost.sum())}")

strategies = {
    "Model: expected £ at risk (p × value)": d["churn_value_at_risk"],
    "Model: churn probability only": d["p_churn"],
    "Rule: longest silence (recency)": d["recency_days"],
    "Rule: biggest spenders": d["value_90d"],
    "Random contact": pd.Series(rng.random(N)),
}
rows = []
for k in (0.05, 0.10, 0.20):
    n = int(N * k)
    for name, score in strategies.items():
        top = np.argsort(-score.values)[:n]
        rows.append({"Budget": f"top {k:.0%} ({n} customers)", "Strategy": name,
                     "Churners reached": int(d["churn_90d"].values[top].sum()),
                     "£ at risk reached": lost[top].sum(),
                     "% of lost £ reached": lost[top].sum() / lost.sum()})
ret = pd.DataFrame(rows)
print(ret.to_string(index=False, formatters={"£ at risk reached": gbp, "% of lost £ reached": pct}))
ret.to_csv(OUT + "retention_targeting.csv", index=False)

# Campaign economics at a 10% budget. No campaign was ever run on this data, so the save rate
# and cost are ASSUMPTIONS, shown as a grid rather than one number.
n10 = int(N * 0.10)
top_model = np.argsort(-d["churn_value_at_risk"].values)[:n10]
top_rule = np.argsort(-d["recency_days"].values)[:n10]
reach_model, reach_rule = lost[top_model].sum(), lost[top_rule].sum()
print(f"\nCampaign on the top 10% ({n10} customers): £ at risk reached - model {gbp(reach_model)} "
      f"vs recency rule {gbp(reach_rule)} ({reach_model / max(reach_rule, 1):.1f}x)")
grid = []
for save in (0.05, 0.10, 0.20):          # share of reached churners the campaign keeps
    for cost in (10, 25, 50):           # £ per contacted customer (call / voucher)
        grid.append({"Save rate": save, "Cost / contact": cost,
                     "Net £ (model)": save * reach_model - cost * n10,
                     "Net £ (recency rule)": save * reach_rule - cost * n10})
grid = pd.DataFrame(grid)
print("Net 90-day £ = save rate × £ reached − cost × contacts")
print(grid.to_string(index=False, formatters={"Save rate": pct, "Cost / contact": gbp,
                                              "Net £ (model)": gbp, "Net £ (recency rule)": gbp}))
grid.to_csv(OUT + "retention_campaign_sensitivity.csv", index=False)


# =============================================================== B. Cancellation policy
section("B. CANCELLATION POLICY: ORDER CONFIRMATION FOR HIGH-RISK ACCOUNTS")
fut_c = d["future_cancel_value_90d"].values
# Expected £ cancelled = P(cancel) × how much an account cancels WHEN it cancels.
# That size scales with the account's spend; the ratio is learned from the TRAINING snapshots only:
#   cancelled £ / typical 90-day spend, among training rows that did cancel.
trf = pd.read_parquet(P + "features.parquet").query("split == 'train' and cancel_90d == 1")
size_ratio = trf["future_cancel_value_90d"].sum() / typical90(trf).sum()
d["cancel_exp_loss"] = d["p_cancel"] * size_ratio * d["value_90d"]
print(f"When an account cancels, it cancels ~{size_ratio:.1%} of its typical 90-day spend (training data)")
print(f"Calibration check: predicted cancelled £ {gbp(d.cancel_exp_loss.sum())} vs actual {gbp(d.future_cancel_value_90d.sum())}")
cstrat = {
    "Model: expected £ loss (p × size)": d["cancel_exp_loss"],
    "Model: cancel probability only": d["p_cancel"],
    "Rule: biggest spenders": d["monetary"],
    "Rule: past cancellers by £": d["cancel_value"],
}
cstrat = {k: pd.Series(np.asarray(v, float)) for k, v in cstrat.items()}
crow = []
for k in (0.05, 0.10):
    n = int(N * k)
    for name, score in cstrat.items():
        top = np.argsort(-score.values)[:n]
        crow.append({"Flag": f"top {k:.0%} ({n} accounts)", "Strategy": name,
                     "Cancelled £ covered": fut_c[top].sum(),
                     "% of cancelled £": fut_c[top].sum() / fut_c.sum(),
                     "% of next-90d revenue touched": d["future_revenue_90d"].values[top].sum() / d["future_revenue_90d"].sum()})
canc = pd.DataFrame(crow)
print(f"Cancelled £ in the test window (customers in the snapshot): {gbp(fut_c.sum())}")
print(canc.to_string(index=False, formatters={"Cancelled £ covered": gbp, "% of cancelled £": pct,
                                               "% of next-90d revenue touched": pct}))
canc.to_csv(OUT + "cancel_policy_targeting.csv", index=False)
best5 = canc[(canc.Flag.str.startswith("top 5%")) & canc.Strategy.str.startswith("Model: expected")].iloc[0]
print("\nIf confirming orders prevents a share of flagged accounts' cancellations (assumption):")
for prevent in (0.2, 0.3, 0.5):
    print(f"  prevent {prevent:.0%} -> £{prevent * best5['Cancelled £ covered']:,.0f} saved per 90 days "
          f"(about £{4 * prevent * best5['Cancelled £ covered']:,.0f} a year at the same rate)")


# =============================================================== C. Process controls
section("C. PROCESS CONTROLS (rule-based, from the Step 2 audit)")
sales = pd.read_parquet(P + "sales.parquet")
nonprod = pd.read_parquet(P + "non_product.parquet")
wo = pd.read_csv(S2 + "writeoff_reasons.csv", index_col=0)
rate = pd.read_csv(S2 + "product_cancel_rate.csv", index_col=0)

# C1. Manual credits: a two-person approval above a threshold.
mc = -nonprod.loc[(nonprod.StockCode == "M") & (nonprod.Revenue < 0), "Revenue"]
print("C1 Manual credits - approval threshold options")
for th in (250, 500, 1000):
    big = mc[mc >= th]
    print(f"   >= £{th:<5} {len(big):>3} of {len(mc)} credits ({pct(len(big)/len(mc))}) "
          f"cover {gbp(big.sum())} ({pct(big.sum()/mc.sum())}) of manual credit £")

# C2. Write-offs: mandatory reason codes + packaging review on the worst products.
unexpl = wo.loc[wo.index.isin(["No reason given", "Stock check / unexplained", "Other notes"]), "est_value"].sum()
print(f"\nC2 Write-offs: {gbp(unexpl)} ({pct(unexpl / wo.est_value.sum())}) has no usable reason -> "
      f"mandatory reason codes; damage = {gbp(wo.loc['Damaged','est_value'])} -> packaging review")

# C3. Products with high cancel rates: quality / listing review.
# Use the ACTUAL cancelled £ and a £-based rate. A units-based rate misleads when a cancellation
# refers to an order placed before the data starts (e.g. 9,360 units cancelled at £0.03 each).
cancels = pd.read_parquet(P + "cancellations.parquet")
desc = sales.groupby("StockCode")["Description"].first()
r = pd.DataFrame({"sold_gbp": sales.groupby("StockCode")["Revenue"].sum(),
                  "cancel_gbp": -cancels.groupby("StockCode")["Revenue"].sum()}).fillna(0)
r["rate"] = r["cancel_gbp"] / r["sold_gbp"]
r["desc"] = desc.reindex(r.index)
flag = r[(r["rate"] >= 0.05) & (r["sold_gbp"] >= 1000)].sort_values("cancel_gbp", ascending=False)
print(f"\nC3 Products with >=5% of sales £ cancelled (and >=£1k sold): {len(flag)} of {int((r.sold_gbp >= 1000).sum())} products, "
      f"{gbp(flag.cancel_gbp.sum())} cancelled ({pct(flag.cancel_gbp.sum() / r.cancel_gbp.sum())} of product cancel £)")
print(flag.head(10)[["desc", "sold_gbp", "cancel_gbp", "rate"]]
      .to_string(formatters={"rate": pct, "cancel_gbp": gbp, "sold_gbp": gbp}))
flag.to_csv(OUT + "products_for_review.csv")

# C4. Missing customer IDs: enforce capture (it's worst in peak season).
m = sales.groupby("Month").apply(lambda g: g.loc[~g.HasCustomer, "Revenue"].sum(), include_groups=False)
print(f"\nC4 Revenue with no customer ID: {gbp(m.sum())}; Nov 2011 alone {gbp(m['2011-11'])}")


# =============================================================== D. Customer action list
section("D. CUSTOMER ACTION LIST (scored 10 Sep 2011)")
q_churn = d["churn_value_at_risk"].quantile(0.90)
q_canc = d["cancel_exp_loss"].quantile(0.95)
def action(r):
    if r.cancel_exp_loss >= q_canc and r.churn_value_at_risk >= q_churn:
        return "1. Account review (cancel + churn risk)"
    if r.cancel_exp_loss >= q_canc:
        return "2. Confirm orders before dispatch"
    if r.churn_value_at_risk >= q_churn:
        return "3. Retention contact"
    if r.p_churn >= 0.5:
        return "4. Low-cost win-back email"
    return "5. Business as usual"
d["action"] = d.apply(action, axis=1)
summary = d.groupby("action").agg(customers=("CustomerID", "size"),
                                  churn_risk_gbp=("churn_value_at_risk", "sum"),
                                  cancel_risk_gbp=("cancel_exp_loss", "sum"),
                                  actual_churn=("churn_90d", "mean"),
                                  actual_cancel=("cancel_90d", "mean"))
print(summary.to_string(formatters={"churn_risk_gbp": gbp, "cancel_risk_gbp": gbp,
                                    "actual_churn": pct, "actual_cancel": pct}))
cols = ["CustomerID", "action", "p_churn", "p_cancel", "value_90d", "monetary",
        "churn_value_at_risk", "cancel_exp_loss", "clv_90d", "churn_90d", "cancel_90d",
        "future_revenue_90d", "future_cancel_value_90d"]
d[cols].sort_values(["action", "churn_value_at_risk"], ascending=[True, False]).to_csv(OUT + "customer_action_list.csv", index=False)


# =============================================================== charts
BLUE, GREY, INK, MUTED, SURFACE = "#2a78d6", "#b8b7b0", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})
# Gains curve: share of lost £ reached vs share of customers contacted.
fig, ax = plt.subplots(figsize=(8, 4))
xs = np.arange(1, N + 1) / N * 100
for name, score, color, lw in [("Model: expected £ at risk", strategies["Model: expected £ at risk (p × value)"], BLUE, 2.2),
                               ("Rule: longest silence", strategies["Rule: longest silence (recency)"], GREY, 1.8)]:
    order = np.argsort(-score.values)
    ax.plot(xs, np.cumsum(lost[order]) / lost.sum() * 100, color=color, lw=lw)
    y20 = np.cumsum(lost[order])[int(N * .2) - 1] / lost.sum() * 100
    ax.text(21, y20, f"{name}: {y20:.0f}%", va="center", fontsize=9, color=INK)
ax.plot([0, 100], [0, 100], color=MUTED, lw=0.8, ls="--")
ax.text(60, 52, "random contact", color=MUTED, fontsize=9, rotation=27)
ax.axvline(20, color=MUTED, lw=0.6)
ax.set_xlim(0, 100); ax.set_ylim(0, 101)
ax.set_xlabel("% of customers contacted (highest risk first)")
ax.set_ylabel("% of lost revenue reached")
ax.set_title("Retention targeting: £ reached by contact budget")
fig.tight_layout(); fig.savefig(OUT + "charts/01_retention_gains.png", dpi=160); plt.close(fig)

c5 = canc[canc.Flag.str.startswith("top 5%")].sort_values("% of cancelled £")
fig, ax = plt.subplots(figsize=(8, 2.8))
ax.barh(c5["Strategy"], c5["% of cancelled £"] * 100,
        color=[BLUE if s.startswith("Model: expected") else GREY for s in c5["Strategy"]], height=0.6)
for y_, v in enumerate(c5["% of cancelled £"]):
    ax.text(v * 100 + 0.8, y_, f"{v:.0%}", va="center", fontsize=9)
ax.set_xlabel("% of next-90-day cancelled £ covered by flagging 5% of accounts")
ax.set_title("Cancellation policy: which 5% of accounts to flag")
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(OUT + "charts/02_cancel_policy.png", dpi=160); plt.close(fig)
print("\nSaved to", OUT)
