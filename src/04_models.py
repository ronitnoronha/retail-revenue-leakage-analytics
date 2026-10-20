"""
Step 4 - Model comparison.

Problem 1, CHURN (no purchase in the next 90 days) - 7 models + a rule baseline:
  Baseline rule (recency), Logistic Regression, Random Forest, LightGBM, MLP neural network,
  Cox proportional hazards, Random Survival Forest, BG/NBD.
Problem 2, CANCELLATION (any cancellation in the next 90 days) - 4 classifiers + baseline:
  Baseline rule (spend), Logistic Regression, Random Forest, LightGBM, MLP.

Validation
  * Tuning + cross-validation on the 4 training snapshots with GroupKFold by customer
    (a customer is never in both the fitting and validation folds).
  * Final score on the out-of-time test snapshot (10 Sep 2011 -> next 90 days), never touched
    during tuning.
Outputs: outputs/step4/  (tables, test predictions, charts)
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from lightgbm import LGBMClassifier
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
import sys; sys.path.insert(0, "src")
from bgnbd import BGNBD, GammaGamma, rfm_summary
import joblib

warnings.filterwarnings("ignore")
SEED = 42
H = 90
P = "data/processed/"
OUT = "outputs/step4/"
os.makedirs(OUT + "charts", exist_ok=True)
pd.set_option("display.width", 200)

feat = pd.read_parquet(P + "features.parquet")
LABELS = ["churn_90d", "cancel_90d", "future_revenue_90d", "future_cancel_value_90d"]
FEATURES = [c for c in feat.columns if c not in ["CustomerID", "cutoff", "split"] + LABELS]
BINARY = [c for c in FEATURES if c.startswith("country_")]
CONT = [c for c in FEATURES if c not in BINARY]

tr = feat[feat.split == "train"].reset_index(drop=True)
te = feat[feat.split == "test"].reset_index(drop=True)
X_tr, X_te = tr[FEATURES], te[FEATURES]
groups = tr["CustomerID"]
cv = GroupKFold(n_splits=5)

# ---------------------------------------------------------------- survival target
# Days from the cutoff to the customer's next purchase, capped (censored) at 90 days.
# "Event" = they bought again. Churn probability = P(no purchase by day 90) = S(90).
sales = pd.read_parquet(P + "sales.parquet")
sales = sales[sales.HasCustomer]
orders = (sales.groupby(["CustomerID", "Invoice"])
          .agg(date=("InvoiceDate", "min"), value=("Revenue", "sum")).reset_index())

def next_purchase_days(df):
    out = np.full(len(df), float(H))
    for cut, idx in df.groupby("cutoff").groups.items():
        fut = orders[(orders.date >= cut) & (orders.date < cut + pd.Timedelta(days=H))]
        first = fut.groupby("CustomerID")["date"].min()
        d = (df.loc[idx, "CustomerID"].map(first) - cut).dt.total_seconds() / 86400
        out[idx] = d.fillna(H).clip(lower=0.01).values
    return out

tr["t_next"], te["t_next"] = next_purchase_days(tr), next_purchase_days(te)
y_surv_tr = Surv.from_arrays(event=tr["churn_90d"] == 0, time=tr["t_next"])


# ---------------------------------------------------------------- preprocessing
# Linear, neural and Cox models need inputs on similar scales: log1p tames the heavy right
# skew (spend ranges from £3 to £200k+), then StandardScaler gives mean 0 / sd 1.
# Tree models split on thresholds, so they get the raw features.
def scaled():
    return ColumnTransformer([
        ("cont", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), CONT),
        ("bin", "passthrough", BINARY),
    ])


CLASSIFIERS = {
    "Logistic Regression": (
        Pipeline([("prep", scaled()), ("m", LogisticRegression(max_iter=2000))]),
        {"m__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]}),
    "Random Forest": (
        RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=SEED),
        {"max_depth": [6, 10, 14, None], "min_samples_leaf": [5, 10, 20, 40],
         "max_features": ["sqrt", 0.3, 0.5]}),
    "LightGBM": (
        LGBMClassifier(n_estimators=400, random_state=SEED, verbose=-1, n_jobs=4),
        {"learning_rate": [0.01, 0.02, 0.05], "num_leaves": [7, 15, 31],
         "min_child_samples": [20, 50, 100], "subsample": [0.7, 0.9], "subsample_freq": [1],
         "colsample_bytree": [0.6, 0.8, 1.0], "reg_lambda": [0, 1, 5]}),
    "MLP Neural Net": (
        Pipeline([("prep", scaled()),
                  ("m", MLPClassifier(max_iter=500, early_stopping=True, random_state=SEED))]),
        {"m__hidden_layer_sizes": [(32,), (64, 32), (128, 64), (64, 32, 16)],
         "m__alpha": [1e-4, 1e-3, 1e-2, 1e-1], "m__learning_rate_init": [1e-3, 3e-3]}),
}


def top_k_capture(y, score, k=0.2, weight=None):
    """Share of all positives (or of their £ weight) found in the top k% of scores."""
    w = np.ones(len(y)) if weight is None else np.asarray(weight)
    order = np.argsort(-np.asarray(score))[: int(len(y) * k)]
    pos = np.asarray(y) == 1
    return (w[order] * pos[order]).sum() / (w * pos).sum()


def evaluate(y, p, name, fit_s, cv_auc=None, cv_std=None, cv_ap=None, is_prob=True, value=None):
    row = {"Model": name,
           "CV ROC-AUC": cv_auc, "CV ± sd": cv_std, "CV PR-AUC": cv_ap,
           "Test ROC-AUC": roc_auc_score(y, p), "Test PR-AUC": average_precision_score(y, p),
           "Test Brier": brier_score_loss(y, p) if is_prob else np.nan,
           "Top-20% capture": top_k_capture(y, p),
           "Fit time (s)": fit_s}
    if value is not None:
        row["Top-20% £ capture"] = top_k_capture(y, p, weight=value)
    return row


def run_classifiers(target, scoring, value_col=None):
    rows, preds, fitted = [], {}, {}
    y_tr, y_te = tr[target], te[target]
    for name, (est, grid) in CLASSIFIERS.items():
        t0 = time.time()
        search = RandomizedSearchCV(est, grid, n_iter=min(15, np.prod([len(v) for v in grid.values()])),
                                    scoring={"auc": "roc_auc", "ap": "average_precision"}, refit=scoring,
                                    cv=cv, n_jobs=1 if name != "Random Forest" else 1,
                                    random_state=SEED)
        search.fit(X_tr, y_tr, groups=groups)
        i = search.best_index_
        cvr = search.cv_results_
        best = search.best_estimator_                         # refit on all training rows
        p = best.predict_proba(X_te)[:, 1]
        fit_s = time.time() - t0
        preds[name], fitted[name] = p, best
        rows.append(evaluate(y_te, p, name, fit_s, cvr["mean_test_auc"][i], cvr["std_test_auc"][i],
                             cvr["mean_test_ap"][i], value=te[value_col] if value_col else None))
        print(f"  {name:<22} test AUC {rows[-1]['Test ROC-AUC']:.3f}   best params {search.best_params_}")
    return rows, preds, fitted


def cv_manual(make_and_score):
    """GroupKFold CV for models that don't fit the sklearn classifier interface."""
    aucs, aps = [], []
    for a, b in cv.split(X_tr, tr["churn_90d"], groups):
        p = make_and_score(a, b)
        aucs.append(roc_auc_score(tr["churn_90d"].iloc[b], p))
        aps.append(average_precision_score(tr["churn_90d"].iloc[b], p))
    return np.mean(aucs), np.std(aucs), np.mean(aps)


def surv_churn_prob(model, X):
    """S(t) just before day 90 = probability of no purchase in the window."""
    fns = model.predict_survival_function(X)
    return np.array([fn(min(H - 1e-6, fn.x[-1])) for fn in fns])


# =================================================================== CHURN
print("\n=== CHURN: no purchase in the next 90 days ===")
churn_rows, churn_pred, churn_models = run_classifiers("churn_90d", "auc")
y_te = te["churn_90d"]

# Baseline: rank by recency (days since last order) - what an analyst would do in Excel.
cvb = cv_manual(lambda a, b: tr["recency_days"].iloc[b].values)
churn_rows.insert(0, evaluate(y_te, te["recency_days"], "Baseline: recency rule", 0, *cvb, is_prob=False))
churn_pred["Baseline: recency rule"] = te["recency_days"].values

# Cox proportional hazards (ridge-penalised for stability)
prep = scaled().fit(X_tr)
Xs_tr, Xs_te = prep.transform(X_tr), prep.transform(X_te)
def cox_fold(a, b):
    pp = scaled().fit(X_tr.iloc[a])
    m = CoxPHSurvivalAnalysis(alpha=1.0).fit(pp.transform(X_tr.iloc[a]), y_surv_tr[a])
    return surv_churn_prob(m, pp.transform(X_tr.iloc[b]))
cvc = cv_manual(cox_fold)
t0 = time.time()
cox = CoxPHSurvivalAnalysis(alpha=1.0).fit(Xs_tr, y_surv_tr)
p = surv_churn_prob(cox, Xs_te)
churn_rows.append(evaluate(y_te, p, "Cox Proportional Hazards", time.time() - t0, *cvc))
churn_pred["Cox Proportional Hazards"] = p
print(f"  {'Cox Proportional Hazards':<22} test AUC {churn_rows[-1]['Test ROC-AUC']:.3f}")

# Random Survival Forest
rsf_params = dict(n_estimators=200, min_samples_leaf=20, max_features="sqrt", n_jobs=-1, random_state=SEED)
def rsf_fold(a, b):
    m = RandomSurvivalForest(**rsf_params).fit(X_tr.iloc[a], y_surv_tr[a])
    return surv_churn_prob(m, X_tr.iloc[b])
cvr_ = cv_manual(rsf_fold)
t0 = time.time()
rsf = RandomSurvivalForest(**rsf_params).fit(X_tr, y_surv_tr)
p = surv_churn_prob(rsf, X_te)
churn_rows.append(evaluate(y_te, p, "Random Survival Forest", time.time() - t0, *cvr_))
churn_pred["Random Survival Forest"] = p
print(f"  {'Random Survival Forest':<22} test AUC {churn_rows[-1]['Test ROC-AUC']:.3f}")

# BG/NBD: unsupervised - fitted on each cutoff's purchase history, no labels used.
# Its "CV" score is the average over the 4 training snapshots (each fitted only on its own past).
def bgnbd_at(cut, ids):
    s = rfm_summary(orders, cut)
    m = BGNBD().fit(s.x, s.t_x, s["T"])
    s = s.reindex(ids)
    return m, m.p_no_purchase(H, s.x, s.t_x, s["T"]), s
auc_s, ap_s = [], []
for cut, d in tr.groupby("cutoff"):
    _, p, _ = bgnbd_at(cut, d["CustomerID"].values)
    auc_s.append(roc_auc_score(d["churn_90d"], p)); ap_s.append(average_precision_score(d["churn_90d"], p))
t0 = time.time()
bg, p, bg_summary = bgnbd_at(te["cutoff"].iloc[0], te["CustomerID"].values)
churn_rows.append(evaluate(y_te, p, "BG/NBD", time.time() - t0, np.mean(auc_s), np.std(auc_s), np.mean(ap_s)))
churn_pred["BG/NBD"] = np.asarray(p)
print(f"  {'BG/NBD':<22} test AUC {churn_rows[-1]['Test ROC-AUC']:.3f}   "
      f"params r={bg.r:.3f} alpha={bg.alpha:.1f} a={bg.a:.3f} b={bg.b:.3f}")

churn_tbl = pd.DataFrame(churn_rows).sort_values("Test ROC-AUC", ascending=False)


# =================================================================== CANCELLATION
print("\n=== CANCELLATION: any cancellation in the next 90 days ===")
canc_rows, canc_pred, canc_models = run_classifiers("cancel_90d", "ap", value_col="future_cancel_value_90d")
y_te_c = te["cancel_90d"]
cvb = []
for a, b in cv.split(X_tr, tr["cancel_90d"], groups):
    yb = tr["cancel_90d"].iloc[b]; sb = tr["monetary"].iloc[b]
    cvb.append((roc_auc_score(yb, sb), average_precision_score(yb, sb)))
cvb = np.array(cvb)
canc_rows.insert(0, evaluate(y_te_c, te["monetary"], "Baseline: spend rule", 0, cvb[:, 0].mean(),
                             cvb[:, 0].std(), cvb[:, 1].mean(), is_prob=False,
                             value=te["future_cancel_value_90d"]))
canc_pred["Baseline: spend rule"] = te["monetary"].values
canc_tbl = pd.DataFrame(canc_rows).sort_values("Test PR-AUC", ascending=False)


# =================================================================== report
fmt = {c: "{:.3f}".format for c in ["CV ROC-AUC", "CV ± sd", "CV PR-AUC", "Test ROC-AUC", "Test PR-AUC", "Test Brier"]}
fmt.update({"Top-20% capture": "{:.1%}".format, "Top-20% £ capture": "{:.1%}".format, "Fit time (s)": "{:.1f}".format})
print(f"\nCHURN results (test base rate {y_te.mean():.1%}, n={len(te):,})")
print(churn_tbl.to_string(index=False, formatters=fmt, na_rep="–"))
print(f"\nCANCELLATION results (test base rate {y_te_c.mean():.1%})")
print(canc_tbl.to_string(index=False, formatters=fmt, na_rep="–"))
churn_tbl.to_csv(OUT + "churn_model_comparison.csv", index=False)
canc_tbl.to_csv(OUT + "cancel_model_comparison.csv", index=False)

# Test predictions for every model (used in Step 5)
pred = te[["CustomerID", "cutoff"] + LABELS + ["monetary", "revenue_90d"]].copy()
for k, v in churn_pred.items():
    pred["churn|" + k] = v
for k, v in canc_pred.items():
    pred["cancel|" + k] = v
pred["bg_exp_purchases_90d"] = np.asarray(bg.expected_purchases(H, bg_summary.x, bg_summary.t_x, bg_summary["T"]))
# Gamma-Gamma: expected value per purchase -> 90-day CLV = expected purchases x expected value
gg = GammaGamma().fit(bg_summary.x, bg_summary.m)
pred["gg_exp_order_value"] = np.asarray(gg.expected_value(bg_summary.x, bg_summary.m))
pred["clv_90d"] = pred["bg_exp_purchases_90d"] * pred["gg_exp_order_value"]
print(f"\nGamma-Gamma params p={gg.p:.2f} q={gg.q:.2f} v={gg.v:.1f}; "
      f"90-day CLV predicted £{pred.clv_90d.sum():,.0f} vs actual £{pred.future_revenue_90d.sum():,.0f}; "
      f"Spearman rank corr {pred.clv_90d.corr(pred.future_revenue_90d, method='spearman'):.2f}")
pred.to_parquet(OUT + "test_predictions.parquet", index=False)
joblib.dump({"churn": churn_models, "cancel": canc_models, "cox": cox, "features": FEATURES},
            OUT + "models.joblib")


# =================================================================== interpretation
best_churn = churn_tbl[churn_tbl.Model.isin(CLASSIFIERS)].iloc[0]["Model"]
best_canc = canc_tbl[canc_tbl.Model.isin(CLASSIFIERS)].iloc[0]["Model"]
imp = {}
for label, name, model, y, scoring in [("churn", best_churn, churn_models[best_churn], y_te, "roc_auc"),
                                       ("cancel", best_canc, canc_models[best_canc], y_te_c, "average_precision")]:
    r = permutation_importance(model, X_te, y, scoring=scoring, n_repeats=10, random_state=SEED, n_jobs=-1)
    imp[label] = pd.Series(r.importances_mean, index=FEATURES).sort_values(ascending=False)
    print(f"\nPermutation importance - {label} ({name}): drop in {scoring} when a feature is shuffled")
    print(imp[label].head(10).round(4).to_string())
    imp[label].to_csv(OUT + f"importance_{label}.csv")

lr = churn_models["Logistic Regression"].named_steps["m"]
coef = pd.Series(lr.coef_[0], index=CONT + BINARY).sort_values()
print("\nLogistic Regression (churn) coefficients on standardised log features - largest effects")
print(pd.concat([coef.head(5), coef.tail(5)]).round(3).to_string())


# =================================================================== charts
BLUE, GREY, INK, MUTED, SURFACE = "#2a78d6", "#b8b7b0", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

def bars(tbl, metric, title, fname, lo):
    d = tbl.sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(d) + 1.2))
    colors = [GREY if m.startswith("Baseline") else BLUE for m in d["Model"]]
    ax.barh(d["Model"], d[metric] - lo, left=lo, color=colors, height=0.6)
    for y_, v in enumerate(d[metric]):
        ax.text(v + 0.003, y_, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(lo, d[metric].max() + 0.04)
    ax.set_xlabel(f"{metric} (out-of-time test, 10 Sep - 9 Dec 2011)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(OUT + "charts/" + fname, dpi=160); plt.close(fig)

bars(churn_tbl, "Test ROC-AUC", "Churn: 7 models vs a recency rule (grey)", "01_churn_auc.png", 0.6)
bars(canc_tbl, "Test PR-AUC", "Cancellation: 4 models vs a spend rule (grey)", "02_cancel_prauc.png",
     round(y_te_c.mean(), 2) - 0.05)

for label, name in [("churn", best_churn), ("cancel", best_canc)]:
    s = imp[label].head(10).sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(s.index, s.values, color=BLUE, height=0.6)
    ax.set_xlabel("drop in score when the feature is shuffled")
    ax.set_title(f"What drives the {label} prediction ({name})")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(OUT + f"charts/03_importance_{label}.png", dpi=160); plt.close(fig)

print("\nSaved to", OUT)
