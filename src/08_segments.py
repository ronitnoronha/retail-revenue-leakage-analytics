"""
Step 8 - Customer segmentation (clustering) for targeted marketing.

Who: the 2,783 customers active on 10 Sep 2011 (bought in the previous 180 days) - the same
snapshot the churn model scored, so segments can be checked against what customers actually
did over the next 90 days (10 Sep - 9 Dec 2011).
Features: behaviour known on 10 Sep only (built in Step 3, no future information).

1. Choose features and transform them (log for skew, then standardise)
2. Compare 3 algorithms over k = 2..10: K-Means, Gaussian Mixture, Agglomerative (Ward)
3. Pick k from the metrics plus business usefulness; check stability across random seeds
4. Profile and name each segment; validate on the next 90 days; assign a marketing action
Outputs: outputs/step8/
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

P, S5, OUT = "data/processed/", "outputs/step5/", "outputs/step8/"
os.makedirs(OUT + "charts", exist_ok=True)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
SEED = 42
K = int(sys.argv[1]) if len(sys.argv) > 1 else None     # chosen number of segments (after exploring)

feat = pd.read_parquet(P + "features.parquet").query("split == 'test'").reset_index(drop=True)

# ---------------------------------------------------------------- 1. features
# One feature per behaviour, avoiding near-duplicates (e.g. orders_90d mostly repeats frequency):
CLUSTER_FEATURES = {
    "recency_days": "days since last order",
    "frequency": "number of orders",
    "monetary": "total spend (£)",
    "avg_order_value": "average order value (£)",
    "tenure_days": "days since first order",
    "avg_units_per_order": "units per order (wholesale size)",
    "n_products": "distinct products bought",
    "avg_unit_price": "average unit price (£)",
    "revenue_trend": "last 90d vs previous 90d spend",
    "cancel_rate": "cancelled £ / spend",
}
X_raw = feat[list(CLUSTER_FEATURES)].copy()
# log1p: spend runs from £3 to £200k+; without it K-Means would be driven by a handful of giants.
X_log = np.log1p(X_raw.clip(lower=0))
X = StandardScaler().fit_transform(X_log)
print(f"Customers: {len(X):,}   features: {len(CLUSTER_FEATURES)}")
print("Correlation check (largest absolute pairs after log):")
c = pd.DataFrame(X, columns=list(CLUSTER_FEATURES)).corr().abs()
pairs = c.where(np.triu(np.ones(c.shape, bool), 1)).stack().sort_values(ascending=False)
print(pairs.head(4).round(2).to_string())

# ---------------------------------------------------------------- 2. compare algorithms
rows = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
    gm = GaussianMixture(n_components=k, covariance_type="full", n_init=3, random_state=SEED).fit(X)
    ag = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
    for name, labels in [("K-Means", km.labels_), ("Gaussian Mixture", gm.predict(X)), ("Agglomerative (Ward)", ag.labels_)]:
        sizes = np.bincount(labels, minlength=k)
        rows.append({"k": k, "algorithm": name,
                     "silhouette": silhouette_score(X, labels, random_state=SEED, sample_size=None),
                     "davies_bouldin": davies_bouldin_score(X, labels),
                     "calinski_harabasz": calinski_harabasz_score(X, labels),
                     "smallest_segment_pct": sizes.min() / len(labels),
                     "inertia": km.inertia_ if name == "K-Means" else np.nan,
                     "gmm_bic": gm.bic(X) if name == "Gaussian Mixture" else np.nan})
sel = pd.DataFrame(rows)
sel.to_csv(OUT + "model_selection.csv", index=False)
print("\nSilhouette (higher = better separated) by k")
print(sel.pivot(index="k", columns="algorithm", values="silhouette").round(3).to_string())
print("\nDavies-Bouldin (lower = better) by k")
print(sel.pivot(index="k", columns="algorithm", values="davies_bouldin").round(3).to_string())
print("\nSmallest segment share by k")
print(sel.pivot(index="k", columns="algorithm", values="smallest_segment_pct").round(3).to_string())
km_rows = sel[sel.algorithm == "K-Means"].set_index("k")
print("\nK-Means inertia (elbow) and GMM BIC (lower = better)")
print(pd.DataFrame({"inertia": km_rows["inertia"],
                    "gmm_bic": sel[sel.algorithm == "Gaussian Mixture"].set_index("k")["gmm_bic"]}).round(0).to_string())

if K is None:
    sys.exit("\nRun again with the chosen k, e.g. python src/08_segments.py 5")

# ---------------------------------------------------------------- 3. final model + stability
km = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit(X)
labels = km.labels_
# Stability: do different random starts, and models fitted on 80% subsamples, find the same groups?
# Adjusted Rand Index (ARI): 1 = identical grouping, 0 = no better than chance.
ari_seed = [adjusted_rand_score(labels, KMeans(n_clusters=K, n_init=20, random_state=s).fit_predict(X)) for s in range(1, 11)]
rng = np.random.default_rng(SEED)
ari_boot = []
for _ in range(20):
    idx = rng.choice(len(X), int(0.8 * len(X)), replace=False)
    sub = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit(X[idx])
    ari_boot.append(adjusted_rand_score(labels, sub.predict(X)))
ari_ward = adjusted_rand_score(labels, AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(X))
print(f"\nChosen: K-Means, k={K}")
print(f"Stability ARI - 10 other seeds: mean {np.mean(ari_seed):.3f} (min {np.min(ari_seed):.3f}); "
      f"20 subsamples of 80%: mean {np.mean(ari_boot):.3f} (min {np.min(ari_boot):.3f}); "
      f"agreement with Ward clustering: {ari_ward:.3f}")

# ---------------------------------------------------------------- 4. profile and name
seg = feat.assign(segment_id=labels)
g = seg.groupby("segment_id")
prof = g[list(CLUSTER_FEATURES)].median()
prof["customers"] = g.size()
prof["customer_share"] = prof["customers"] / len(seg)
prof["spend_share"] = g["monetary"].sum() / seg["monetary"].sum()

# Name segments from their profile (rules on the medians, so names follow the data, not label numbers)
names = {}
left = set(prof.index)
def take(idx, name):
    names[idx] = name; left.discard(idx)
take(seg.groupby("segment_id")["cancel_rate"].mean().idxmax(), "Serial cancellers")
take(prof.loc[list(left), "monetary"].idxmax(), "Loyal wholesale core")
take(prof.loc[list(left), "tenure_days"].idxmin(), "New customers")
rest = prof.loc[list(left)].sort_values("monetary")
for i, idx in enumerate(rest.index):
    take(idx, ["Lapsing small buyers", "Regular mid-tier"][i] if len(rest) == 2 else f"Segment {idx}")
seg["segment"] = seg["segment_id"].map(names)
prof.index = prof.index.map(names)

ACTION = {
    "Loyal wholesale core": "Protect: account manager, early access to new ranges, volume pricing. "
                            "Handle their cancellations through the account manager, not checkout friction.",
    "Regular mid-tier": "Re-engage: personalised reorder reminders around their usual buying gap, "
                        "bundles of products they already buy. Main pool for the retention campaign.",
    "New customers": "Onboard: second-order offer within ~30 days (the median gap between orders is 28 days), "
                     "welcome series showing best-sellers.",
    "Lapsing small buyers": "Win back cheaply: automated email with a time-limited discount. No phone calls - "
                            "low value per customer.",
    "Serial cancellers": "Order confirmation (or deposit) before dispatch; review why they cancel "
                         "(product, delivery, pricing).",
}

# ---------------------------------------------------------------- 5. validate on the next 90 days
al = pd.read_csv(S5 + "customer_action_list.csv", dtype={"CustomerID": str})
seg = seg.merge(al[["CustomerID", "p_churn", "churn_value_at_risk"]], on="CustomerID", how="left")
g = seg.groupby("segment")
out = pd.DataFrame({
    "customers": g.size(),
    "customer_share": g.size() / len(seg),
    "spend_share": g["monetary"].sum() / seg["monetary"].sum(),
    "median_orders": g["frequency"].median(),
    "median_spend_gbp": g["monetary"].median(),
    "median_recency_days": g["recency_days"].median(),
    "median_units_per_order": g["avg_units_per_order"].median(),
    "past_cancel_rate": g["cancel_value"].sum() / g["monetary"].sum(),
    "predicted_churn": g["p_churn"].mean(),
    "actual_churn_90d": g["churn_90d"].mean(),
    "actual_cancel_90d": g["cancel_90d"].mean(),
    "next_90d_revenue_gbp": g["future_revenue_90d"].sum(),
    "churn_gbp_at_risk": g["churn_value_at_risk"].sum(),
}).sort_values("spend_share", ascending=False)
out["action"] = out.index.map(ACTION)
out.to_csv(OUT + "segment_profiles.csv")
fmt = {c: "{:.1%}".format for c in ["customer_share", "spend_share", "past_cancel_rate", "predicted_churn",
                                     "actual_churn_90d", "actual_cancel_90d"]}
fmt.update({c: "£{:,.0f}".format for c in ["median_spend_gbp", "next_90d_revenue_gbp", "churn_gbp_at_risk"]})
print("\nSegment profiles (features as of 10 Sep 2011; 'actual' = what happened 10 Sep - 9 Dec)")
print(out.drop(columns="action").to_string(formatters=fmt))

from scipy.stats import chi2_contingency
tab = pd.crosstab(seg["segment"], seg["churn_90d"])
chi = chi2_contingency(tab)
print(f"\nDo segments differ in actual churn? chi-square = {chi.statistic:.0f}, p = {chi.pvalue:.1e}")
print(f"Overall actual churn {seg.churn_90d.mean():.1%}")
seg[["CustomerID", "segment"] + list(CLUSTER_FEATURES) + ["p_churn", "churn_90d", "cancel_90d", "future_revenue_90d"]] \
    .to_csv(OUT + "customer_segments.csv", index=False)

# ---------------------------------------------------------------- charts
BLUE, INK, MUTED, SURFACE, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})
order = list(out.index)                      # by spend share, fixed colour per segment
color = {s: CAT[i] for i, s in enumerate(order)}

km_sel = sel[sel.algorithm == "K-Means"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, col, title, better in [(axes[0], "silhouette", "Silhouette (higher is better)", max),
                               (axes[1], "davies_bouldin", "Davies-Bouldin (lower is better)", min)]:
    ax.plot(km_sel["k"], km_sel[col], color=BLUE, lw=2, marker="o", ms=4)
    ax.scatter([K], km_sel.loc[km_sel.k == K, col], s=70, facecolor="none", edgecolor=INK, lw=1.5, zorder=3)
    ax.set_title(title, fontsize=10); ax.set_xlabel("number of segments (k)")
fig.suptitle(f"Choosing k for K-Means (circled: k={K})", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT + "charts/01_choosing_k.png", dpi=160); plt.close(fig)

pca = PCA(n_components=2, random_state=SEED).fit(X)
Z = pca.transform(X)
fig, ax = plt.subplots(figsize=(8, 5))
for s in order:
    m = (seg["segment"] == s).values
    ax.scatter(Z[m, 0], Z[m, 1], s=9, alpha=0.6, color=color[s], label=f"{s} ({m.sum():,})", edgecolor="none")
ax.set_xlabel(f"component 1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
ax.set_ylabel(f"component 2 ({pca.explained_variance_ratio_[1]:.0%} of variance)")
ax.set_title("Customer segments, projected to 2 dimensions (PCA)")
ax.legend(frameon=False, fontsize=9, loc="best", markerscale=2)
fig.tight_layout(); fig.savefig(OUT + "charts/02_segments_pca.png", dpi=160); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
y = np.arange(len(order))
axes[0].barh(y - 0.2, out["customer_share"] * 100, height=0.38, color="#b8b7b0", label="share of customers")
axes[0].barh(y + 0.2, out["spend_share"] * 100, height=0.38, color=BLUE, label="share of spend")
axes[0].set_yticks(y, order); axes[0].invert_yaxis()
axes[0].set_xlabel("%"); axes[0].legend(frameon=False, fontsize=8, loc="lower right")
axes[0].set_title("Size vs value", fontsize=10)
axes[1].barh(y, out["actual_churn_90d"] * 100, height=0.6, color=[color[s] for s in order])
axes[1].axvline(seg.churn_90d.mean() * 100, color=MUTED, ls="--", lw=1)
axes[1].text(seg.churn_90d.mean() * 100 + 1, -0.55, f"all customers {seg.churn_90d.mean():.0%}", fontsize=8, color=MUTED)
for i, v in enumerate(out["actual_churn_90d"]):
    axes[1].text(v * 100 + 1, i, f"{v:.0%}", va="center", fontsize=9)
axes[1].set_xlabel("% that stopped buying in the next 90 days"); axes[1].set_title("Actual churn, 10 Sep - 9 Dec 2011", fontsize=10)
for ax in axes: ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(OUT + "charts/03_segment_value_churn.png", dpi=160); plt.close(fig)
print("\nSaved to", OUT)
