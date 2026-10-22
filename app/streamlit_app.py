"""
Retail Leakage & Customer Risk - Streamlit app.

Interactive front end for the project's results (UCI Online Retail II, 2010-2011).
All figures come from the small CSVs in app/data/, produced by the analysis pipeline
(src/01_clean.py ... src/08_segments.py, then app/build_app_data.py).

Run locally:  streamlit run app/streamlit_app.py
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "data"
REPO = "https://github.com/ronitnoronha/retail-revenue-leakage-analytics"

st.set_page_config(page_title="Retail Leakage & Customer Risk", page_icon="📊", layout="wide")

# One colour per meaning, used on every chart
ACCENT, MUTED, REF = "#2a78d6", "#b8b7b0", "#8a8983"
SEGMENTS = ["Loyal wholesale core", "Regular mid-tier", "New customers", "Lapsing small buyers", "Serial cancellers"]
SEG_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
seg_scale = alt.Scale(domain=SEGMENTS, range=SEG_COLORS)


@st.cache_data
def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv", dtype={"customer_id": str, "stock_code": str})


def gbp(v: float) -> str:
    return f"£{v:,.0f}"


def gbp_short(v: float) -> str:
    a = abs(v)
    s = f"£{a / 1e6:.2f}M" if a >= 1e6 else f"£{a / 1e3:.0f}k" if a >= 1e4 else f"£{a / 1e3:.1f}k" if a >= 1e3 else f"£{a:,.0f}"
    return ("−" if v < 0 else "") + s


def pct(v: float, d: int = 1) -> str:
    return f"{v:.{d}%}"


overview = load("overview").iloc[0]
leaks = load("leaks")
monthly = load("monthly")
writeoffs = load("writeoff_reasons")
manual = load("manual_thresholds")
products = load("products_review")
gains = load("retention_gains")
churn_models = load("churn_models")
cancel_models = load("cancel_models")
policy = load("cancel_policy")
segments = load("segment_profiles")
customers = load("customers")

monthly["month_date"] = pd.to_datetime(monthly["month"] + "-01")
monthly["label"] = np.where(monthly["partial_month"], "Dec 2011 (9 days)", "Full month")
leak_total = leaks["value_gbp"].sum()
at10 = gains.loc[(gains["pct_contacted"] - 0.10).abs().idxmin()]

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### About this project")
    st.markdown(
        "A UK online gift retailer sold **£10M** in a year. This app shows where value leaks out between "
        "order and cash, which customers are about to stop buying, and what to do about each.\n\n"
        "**Data:** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), "
        "Dec 2010 – Dec 2011, 541,910 transactions.\n\n"
        "**Method:** data audit and cleaning → process audit → time-based features → 7 churn models compared "
        "on an out-of-time test → value-based targeting → K-Means customer segments.\n\n"
        f"**Code:** [GitHub]({REPO})"
    )
    st.caption("Built by Ronit Noronha · Python, pandas, scikit-learn, LightGBM, scikit-survival, Streamlit")

# ------------------------------------------------------------------ header
st.title("Retail leakage & customer risk")
st.markdown(
    f"Up to **{gbp_short(leak_total)}** (**{pct(leak_total / overview.product_sales_gbp)}** of sales) leaks out between "
    f"order and cash, mostly through internal processes. Contacting the riskiest **10%** of customers by revenue at risk "
    f"reaches **{pct(at10.model_expected_gbp)}** of the revenue churners take with them."
)

tabs = st.tabs(["Overview", "Process audit", "Customer risk", "Segments", "Customer lookup", "Model evidence", "Method"])

# ================================================================== OVERVIEW
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Product sales", gbp_short(overview.product_sales_gbp), "Dec 2010 – Dec 2011", delta_color="off", delta_arrow="off")
    c[1].metric("Value leakage (upper bound)", gbp_short(leak_total), f"{pct(leak_total / overview.product_sales_gbp)} of sales", delta_color="off", delta_arrow="off")
    c[2].metric("Cancellations", pct(overview.clean_cancel_pct), f"raw data said {pct(overview.raw_cancel_pct)}", delta_color="off", delta_arrow="off")
    c[3].metric("Revenue with no customer ID", pct(overview.unattributed_pct), f"{gbp_short(overview.unattributed_gbp)} untraceable", delta_color="off", delta_arrow="off")
    c[4].metric("Customers", f"{int(overview.customers):,}", f"{int(overview.countries)} countries", delta_color="off", delta_arrow="off")

    left, right = st.columns(2)
    with left:
        st.subheader("Where value leaks out")
        st.caption("Internal processes (blue) outweigh customer cancellations. Write-offs are valued at selling price "
                   "(an upper bound); Amazon fees are a channel cost, not an error.")
        d = leaks.assign(group=np.where(leaks["kind"] == "Process", "Internal process", "Other"),
                         value_label=leaks["value_gbp"].map(gbp_short))
        base = alt.Chart(d).encode(y=alt.Y("leak:N", sort=d.sort_values("value_gbp", ascending=False)["leak"].tolist(), title=None, axis=alt.Axis(labelLimit=260)))
        bars = base.mark_bar(cornerRadiusEnd=4, height=22).encode(
            x=alt.X("value_gbp:Q", title="£", axis=alt.Axis(format="~s")),
            color=alt.Color("group:N", scale=alt.Scale(domain=["Internal process", "Other"], range=[ACCENT, MUTED]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("leak:N", title="Leak"), alt.Tooltip("value_gbp:Q", title="£", format=",.0f"),
                     alt.Tooltip("pct_of_sales:Q", title="Share of sales", format=".1%"),
                     alt.Tooltip("root_cause:N", title="Root cause")])
        text = base.mark_text(align="left", dx=4).encode(x="value_gbp:Q", text="value_label:N")
        st.altair_chart((bars + text).properties(height=260), width="stretch")
    with right:
        st.subheader("Monthly product sales")
        st.caption("Sep–Nov is the peak season. December 2011 has only 9 days of data.")
        ch = alt.Chart(monthly).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("yearmonth(month_date):O", title=None, axis=alt.Axis(format="%b %y", labelAngle=0)),
            y=alt.Y("sales_gbp:Q", title="£", axis=alt.Axis(format="~s")),
            color=alt.Color("label:N", scale=alt.Scale(domain=["Full month", "Dec 2011 (9 days)"], range=[ACCENT, MUTED]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("yearmonth(month_date):T", title="Month"),
                     alt.Tooltip("sales_gbp:Q", title="Sales £", format=",.0f"),
                     alt.Tooltip("cancel_rate:Q", title="Cancel rate", format=".1%"),
                     alt.Tooltip("active_customers:Q", title="Active customers", format=",")])
        st.altair_chart(ch.properties(height=260), width="stretch")

# ================================================================== PROCESS AUDIT
with tabs[1]:
    unexpl = writeoffs.loc[~writeoffs["reason_recorded"], "value_gbp"].sum()
    m500 = manual.set_index("threshold_gbp").loc[500]
    nov = monthly.set_index("month").loc["2011-11"]
    c = st.columns(3)
    c[0].metric("Write-off value with no usable reason", pct(unexpl / writeoffs["value_gbp"].sum()),
                f"{gbp_short(unexpl)} at selling price", delta_color="off", delta_arrow="off")
    c[1].metric("Manual-credit value a £500 approval rule covers", pct(m500.pct_of_value),
                f"checking only {pct(m500.pct_of_credits)} of credits", delta_color="off", delta_arrow="off")
    c[2].metric("November revenue with no customer ID", pct(nov.unattributed_pct),
                f"{gbp_short(nov.unattributed_gbp)} in the busiest month", delta_color="off", delta_arrow="off")

    left, right = st.columns(2)
    with left:
        st.subheader("Stock write-offs by recorded reason")
        st.caption("Free-text notes grouped into reasons. Blue = no usable reason. Recommendation: mandatory reason codes.")
        d = writeoffs.assign(group=np.where(writeoffs["reason_recorded"], "Reason recorded", "No usable reason"))
        ch = alt.Chart(d).mark_bar(cornerRadiusEnd=4, height=20).encode(
            y=alt.Y("reason:N", sort=d.sort_values("value_gbp", ascending=False)["reason"].tolist(), title=None),
            x=alt.X("value_gbp:Q", title="£ at selling price", axis=alt.Axis(format="~s")),
            color=alt.Color("group:N", scale=alt.Scale(domain=["No usable reason", "Reason recorded"], range=[ACCENT, MUTED]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("reason:N", title="Reason"), alt.Tooltip("value_gbp:Q", title="£", format=",.0f"),
                     alt.Tooltip("share:Q", title="Share", format=".1%"), alt.Tooltip("lines:Q", title="Lines")])
        st.altair_chart(ch.properties(height=280), width="stretch")
    with right:
        st.subheader("Revenue recorded with no customer ID")
        st.caption("Data capture breaks down when volume is highest. Recommendation: make customer ID mandatory at checkout.")
        line = alt.Chart(monthly).mark_line(point=True, strokeWidth=2, color=ACCENT).encode(
            x=alt.X("yearmonth(month_date):O", title=None, axis=alt.Axis(format="%b %y", labelAngle=0)),
            y=alt.Y("unattributed_pct:Q", title="share of revenue", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("yearmonth(month_date):T", title="Month"),
                     alt.Tooltip("unattributed_pct:Q", title="No customer ID", format=".1%"),
                     alt.Tooltip("unattributed_gbp:Q", title="£", format=",.0f")])
        st.altair_chart(line.properties(height=280), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Manual credits: approval threshold options")
        st.caption("Every manual credit is labelled only \"Manual\". A second approval above a threshold covers most "
                   "of the value while checking few credits. Recommended: £500.")
        st.dataframe(
            manual.assign(threshold=manual["threshold_gbp"].map(lambda v: f"£{v:,}"),
                          need=manual["credits_needing_approval"].astype(str) + " of " + manual["total_credits"].astype(str))
            [["threshold", "need", "pct_of_credits", "value_covered_gbp", "pct_of_value"]],
            hide_index=True, width="stretch",
            column_config={"threshold": "Credits of at least", "need": "Need approval",
                           "pct_of_credits": st.column_config.NumberColumn("Share of credits", format="percent"),
                           "value_covered_gbp": st.column_config.NumberColumn("Value covered", format="£%d"),
                           "pct_of_value": st.column_config.ProgressColumn("Share of value", format="percent", min_value=0, max_value=1)})
    with right:
        st.subheader("Products to review for cancellations")
        st.caption("Top 15 by cancelled £ where at least 5% of sales £ was cancelled (and at least £1k sold).")
        st.dataframe(products, hide_index=True, width="stretch", height=300,
                     column_config={"stock_code": "Code", "description": "Product",
                                    "sold_gbp": st.column_config.NumberColumn("Sold", format="£%d"),
                                    "cancelled_gbp": st.column_config.NumberColumn("Cancelled", format="£%d"),
                                    "cancel_rate": st.column_config.ProgressColumn("Cancel rate", format="percent", min_value=0, max_value=1)})

# ================================================================== CUSTOMER RISK
with tabs[2]:
    st.subheader("Who to contact: share of at-risk revenue reached by contact budget")
    st.caption("Customers scored on 10 Sep 2011; outcomes measured over the next 90 days. Expected £ at risk = churn "
               "probability × typical 90-day spend. Dashed line: random contact.")
    budget = st.slider("Contact budget (share of customers, riskiest first)", 1, 50, 10, format="%d%%")
    long = gains.melt("pct_contacted", ["model_expected_gbp", "biggest_spenders", "recency_rule", "random_expected"],
                      var_name="strategy", value_name="reached")
    names = {"model_expected_gbp": "Model: expected £ at risk", "biggest_spenders": "Rule: biggest spenders",
             "recency_rule": "Rule: longest silence", "random_expected": "Random contact"}
    long["strategy"] = long["strategy"].map(names)
    scale = alt.Scale(domain=list(names.values()), range=[ACCENT, "#eb6834", "#1baf7a", REF])
    lines = alt.Chart(long).mark_line(strokeWidth=2.5).encode(
        x=alt.X("pct_contacted:Q", title="share of customers contacted", axis=alt.Axis(format="%")),
        y=alt.Y("reached:Q", title="share of lost revenue reached", axis=alt.Axis(format="%")),
        color=alt.Color("strategy:N", scale=scale, legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.condition(alt.datum.strategy == "Random contact", alt.value([5, 4]), alt.value([1, 0])),
        tooltip=[alt.Tooltip("strategy:N"), alt.Tooltip("pct_contacted:Q", title="Contacted", format=".0%"),
                 alt.Tooltip("reached:Q", title="Lost revenue reached", format=".1%")])
    rule = alt.Chart(pd.DataFrame({"x": [budget / 100]})).mark_rule(color="#52514e").encode(x="x:Q")
    st.altair_chart((lines + rule).properties(height=320), width="stretch")

    # Campaign simulator, computed from the scored customers
    n = len(customers)
    k = int(round(n * budget / 100))
    lost = np.where(customers["actually_churned"] == 1, customers["typical_90d_spend_gbp"], 0.0)
    order = np.argsort(-customers["churn_gbp_at_risk"].to_numpy())[:k]
    order_rule = np.argsort(-customers["days_since_last_order"].to_numpy())[:k]
    reached, reached_rule = lost[order].sum(), lost[order_rule].sum()
    c = st.columns(3)
    c[0].metric(f"Customers contacted ({budget}%)", f"{k:,}")
    c[1].metric("Lost revenue reached: model", gbp_short(reached), f"{pct(reached / lost.sum())} of {gbp_short(lost.sum())}", delta_color="off", delta_arrow="off")
    c[2].metric("Lost revenue reached: silence rule", gbp_short(reached_rule), f"{pct(reached_rule / lost.sum())}", delta_color="off", delta_arrow="off")

    st.markdown("**Campaign simulator** · save rate and cost are assumptions (no campaign data exists); an A/B test would measure them.")
    s1, s2 = st.columns(2)
    save = s1.slider("Save rate: share of reached churners the campaign keeps", 1, 40, 10, format="%d%%") / 100
    cost = s2.slider("Cost per contact (£)", 0, 100, 25)
    net_model, net_rule = save * reached - cost * k, save * reached_rule - cost * k
    c = st.columns(2)
    c[0].metric("Net £ per quarter: model targeting", gbp_short(net_model))
    c[1].metric("Net £ per quarter: silence rule", gbp_short(net_rule))

    st.subheader("Recommended action per customer")
    st.caption("Actions set from the scores on 10 Sep 2011; the last two columns are what each group actually did next.")
    ag = customers.groupby("action").agg(customers=("customer_id", "size"), churn_gbp_at_risk=("churn_gbp_at_risk", "sum"),
                                         churned=("actually_churned", "mean"), cancelled=("actually_cancelled", "mean")).reset_index()
    st.dataframe(ag.sort_values("churn_gbp_at_risk", ascending=False), hide_index=True, width="stretch",
                 column_config={"action": "Action", "customers": "Customers",
                                "churn_gbp_at_risk": st.column_config.NumberColumn("Churn £ at risk", format="£%d"),
                                "churned": st.column_config.NumberColumn("Actually churned", format="percent"),
                                "cancelled": st.column_config.NumberColumn("Actually cancelled", format="percent")})

# ================================================================== SEGMENTS
with tabs[3]:
    st.subheader("Five customer segments (K-Means)")
    st.caption("Built from behaviour known on 10 Sep 2011, then checked against what each segment actually did in the "
               "next 90 days. K-Means was chosen over Gaussian Mixture and Ward clustering; k = 5 had the best "
               "Davies-Bouldin score and gives each segment a distinct action.")
    seg = segments.set_index("segment").reindex(SEGMENTS).reset_index()
    cols = st.columns(5)
    for col, (_, r), color in zip(cols, seg.iterrows(), SEG_COLORS):
        with col:
            st.markdown(f"<div style='border-top:4px solid {color};padding-top:6px'><b>{r.segment}</b></div>", unsafe_allow_html=True)
            st.metric("Customers", f"{int(r.customers):,}", f"{pct(r.customer_share)} of customers", delta_color="off", delta_arrow="off")
            st.metric("Share of spend", pct(r.spend_share))
            st.metric("Actually churned", pct(r.actual_churn_90d, 0))
            st.caption(r.action)

    left, right = st.columns(2)
    with left:
        st.markdown("**Size vs value**")
        d = seg.melt("segment", ["customer_share", "spend_share"], var_name="measure", value_name="share")
        d["measure"] = d["measure"].map({"customer_share": "Share of customers", "spend_share": "Share of spend"})
        ch = alt.Chart(d).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("segment:N", sort=SEGMENTS, title=None), yOffset="measure:N",
            x=alt.X("share:Q", axis=alt.Axis(format="%"), title=None),
            color=alt.Color("measure:N", scale=alt.Scale(range=[MUTED, ACCENT]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["segment", "measure", alt.Tooltip("share:Q", format=".1%")])
        st.altair_chart(ch.properties(height=300), width="stretch")
    with right:
        st.markdown("**Actual churn, 10 Sep – 9 Dec 2011**")
        overall = customers["actually_churned"].mean()
        bars = alt.Chart(seg).mark_bar(cornerRadiusEnd=4, height=24).encode(
            y=alt.Y("segment:N", sort=SEGMENTS, title=None),
            x=alt.X("actual_churn_90d:Q", axis=alt.Axis(format="%"), title="share that stopped buying"),
            color=alt.Color("segment:N", scale=seg_scale, legend=None),
            tooltip=["segment", alt.Tooltip("actual_churn_90d:Q", title="Actual churn", format=".1%"),
                     alt.Tooltip("predicted_churn:Q", title="Model predicted", format=".1%")])
        ref = alt.Chart(pd.DataFrame({"x": [overall]})).mark_rule(strokeDash=[4, 4], color=REF).encode(x="x:Q")
        st.altair_chart((bars + ref).properties(height=300), width="stretch")
        st.caption(f"Dashed line: all customers ({pct(overall, 0)}). Segments differ strongly (chi-square p ≈ 10⁻⁶⁹).")

    st.markdown("**Every customer, by segment** · hover a point for details")
    pick = st.multiselect("Segments", SEGMENTS, default=SEGMENTS)
    d = customers[customers["segment"].isin(pick)]
    pts = alt.Chart(d).mark_circle(size=28, opacity=0.6).encode(
        x=alt.X("days_since_last_order:Q", title="days since last order (on 10 Sep 2011)"),
        y=alt.Y("total_spend_gbp:Q", title="total spend £ (log scale)", scale=alt.Scale(type="log"), axis=alt.Axis(format="~s")),
        color=alt.Color("segment:N", scale=seg_scale, legend=alt.Legend(orient="top", title=None)),
        tooltip=["customer_id", "segment", "country", "orders", alt.Tooltip("total_spend_gbp:Q", format=",.0f"),
                 alt.Tooltip("churn_probability:Q", format=".0%")])
    st.altair_chart(pts.properties(height=380).interactive(), width="stretch")

    with st.expander("Segment profile table"):
        st.dataframe(seg.drop(columns=["action"]), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="percent") for c in
                                    ["customer_share", "spend_share", "past_cancel_rate", "predicted_churn",
                                     "actual_churn_90d", "actual_cancel_90d"]})

# ================================================================== CUSTOMER LOOKUP
with tabs[4]:
    st.subheader("Look up a customer")
    st.caption("Any of the 2,783 customers active on 10 Sep 2011. Scores use only what was known that day; "
               "\"what actually happened\" covers the next 90 days. Customer IDs are anonymous IDs from the public dataset.")
    ids = customers.sort_values("churn_gbp_at_risk", ascending=False)["customer_id"].tolist()
    cid = st.selectbox("Customer ID (sorted by £ at risk, highest first)", ids)
    r = customers.set_index("customer_id").loc[cid]
    color = SEG_COLORS[SEGMENTS.index(r.segment)]
    st.markdown(f"<div style='border-left:5px solid {color};padding-left:10px'><b>{r.segment}</b> · {r.country}<br>"
                f"Recommended action: <b>{r.action}</b></div>", unsafe_allow_html=True)
    st.write("")
    c = st.columns(4)
    c[0].metric("Churn probability", pct(r.churn_probability, 0))
    c[1].metric("Cancel probability", pct(r.cancel_probability, 0))
    c[2].metric("Typical 90-day spend", gbp(r.typical_90d_spend_gbp))
    c[3].metric("Expected £ at risk", gbp(r.churn_gbp_at_risk))
    c = st.columns(4)
    c[0].metric("Orders before 10 Sep", int(r.orders))
    c[1].metric("Days since last order", int(r.days_since_last_order))
    c[2].metric("Total spend", gbp(r.total_spend_gbp))
    c[3].metric("Units per order", f"{r.units_per_order:,.0f}")
    st.markdown("**What actually happened (10 Sep – 9 Dec 2011)**")
    c = st.columns(3)
    c[0].metric("Kept buying?", "No, churned" if r.actually_churned else "Yes")
    c[1].metric("Cancelled an order?", "Yes" if r.actually_cancelled else "No")
    c[2].metric("Revenue in those 90 days", gbp(r.next_90d_revenue_gbp))

# ================================================================== MODEL EVIDENCE
with tabs[5]:
    st.subheader("Churn: 7 models from 6 families vs a simple rule")
    st.caption("ROC-AUC: how often a random churner is ranked above a random non-churner (0.5 = coin flip). Dot: "
               "out-of-time test (10 Sep – 9 Dec 2011). Bar: cross-validation mean ± 1 sd, grouped by customer. "
               "All models land close together, so the simplest, logistic regression, was chosen.")
    d = churn_models.assign(lo=churn_models.cv_roc_auc - churn_models.cv_sd, hi=churn_models.cv_roc_auc + churn_models.cv_sd,
                            kind=np.where(churn_models.family == "Rule", "Rule baseline", "Model"),
                            label=churn_models.model + " · " + churn_models.family)
    order_m = d.sort_values("test_roc_auc", ascending=False)["label"].tolist()
    y = alt.Y("label:N", sort=order_m, title=None, axis=alt.Axis(labelLimit=320))
    rng = alt.Chart(d).mark_rule(strokeWidth=4, color=MUTED).encode(y=y, x=alt.X("lo:Q", scale=alt.Scale(zero=False), title="ROC-AUC"), x2="hi:Q")
    dot = alt.Chart(d).mark_point(filled=True, size=120).encode(
        y=y, x="test_roc_auc:Q",
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Model", "Rule baseline"], range=[ACCENT, REF]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["model", "family", alt.Tooltip("test_roc_auc:Q", title="Test ROC-AUC", format=".3f"),
                 alt.Tooltip("cv_roc_auc:Q", title="CV ROC-AUC", format=".3f"), alt.Tooltip("test_pr_auc:Q", title="Test PR-AUC", format=".3f")])
    st.altair_chart((rng + dot).properties(height=320), width="stretch")
    st.dataframe(churn_models.sort_values("test_roc_auc", ascending=False), hide_index=True, width="stretch",
                 column_config={"top20_capture": st.column_config.NumberColumn("Churners in top 20%", format="percent")})

    left, right = st.columns(2)
    with left:
        st.subheader("Cancellation models")
        st.caption("Models find more cancelling customers than the spend rule, but the spend rule catches slightly more "
                   "cancelled £, because the biggest cancellations come from the biggest buyers.")
        st.dataframe(cancel_models[["model", "test_pr_auc", "top20_capture", "top20_gbp_capture"]].sort_values("test_pr_auc", ascending=False),
                     hide_index=True, width="stretch",
                     column_config={"test_pr_auc": st.column_config.NumberColumn("Test PR-AUC", format="%.3f"),
                                    "top20_capture": st.column_config.NumberColumn("Cancellers in top 20%", format="percent"),
                                    "top20_gbp_capture": st.column_config.NumberColumn("Cancelled £ in top 20%", format="percent")})
    with right:
        st.subheader("Which 5% of accounts to flag")
        st.caption("Flagging past cancellers covers a similar share of cancelled £ to flagging the biggest accounts, "
                   "while adding friction to far less revenue. Recommended.")
        st.dataframe(policy, hide_index=True, width="stretch",
                     column_config={"strategy": "Flag accounts by",
                                    "pct_cancelled_gbp_covered": st.column_config.ProgressColumn("Cancelled £ covered", format="percent", min_value=0, max_value=1),
                                    "pct_revenue_touched": st.column_config.ProgressColumn("Revenue touched", format="percent", min_value=0, max_value=1)})

# ================================================================== METHOD
with tabs[6]:
    st.subheader("How the analysis was done")
    st.markdown(f"""
1. **Audit and cleaning.** Removed 5,268 duplicate rows; separated fees, manual adjustments and stock write-offs
   from product sales; removed keying errors (e.g. 80,995 units ordered and reversed 12 minutes later). Raw
   cancellations of {pct(overview.raw_cancel_pct)} of sales fell to {pct(overview.clean_cancel_pct)} once those were removed.
2. **Process audit.** Sized each leak, found where it concentrates (Pareto analysis), tested differences with
   Mann-Whitney and chi-square tests, and grouped free-text write-off notes into reasons.
3. **Time-based features.** One row per customer per snapshot date; features use only data before the date,
   labels only the 90 days after. Four training snapshots (Mar–Jun 2011), one test snapshot (10 Sep 2011).
4. **Churn models.** Logistic regression, random forest, LightGBM, an MLP neural network, Cox proportional hazards,
   a random survival forest and BG/NBD (implemented from Fader, Hardie & Lee 2005), against a recency rule.
   Customer-grouped cross-validation for tuning; the September test set used once.
5. **Decisions.** Expected £ at risk = churn probability × typical 90-day spend; targeting and campaign
   economics evaluated on the test period, with assumptions shown as ranges.
6. **Segments.** K-Means on 10 behaviour features (log-transformed, standardised), compared with Gaussian Mixture
   and Ward clustering for k = 2–10; stability checked across seeds and subsamples; validated on next-90-day churn.

Full code and write-ups: [{REPO}]({REPO})
""")
