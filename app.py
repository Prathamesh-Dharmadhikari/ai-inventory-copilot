"""AI Inventory Planning Copilot — Streamlit app.

Run locally:  streamlit run app.py
Deploy free:  share.streamlit.io (connect this GitHub repo, add GEMINI_API_KEY under Secrets — free key from aistudio.google.com)
"""
import os

import pandas as pd
import streamlit as st

import altair as alt
import numpy as np

import analytics as an
import copilot
from planning import Params, compute, kpis, load_data

st.set_page_config(page_title="AI Inventory Planning Copilot", page_icon="📦", layout="wide")


# ---------- API key (optional: dashboard works without it) ----------
def get_secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


API_KEY = get_secret("GEMINI_API_KEY")
APP_PASSWORD = get_secret("APP_PASSWORD")  # optional: protects the free daily AI quota
MODEL_PREF = get_secret("GEMINI_MODEL")

# ---------- sidebar: planning parameters ----------
st.sidebar.header("Planning parameters")
p = Params()
p.holding_rate = st.sidebar.slider("Holding cost rate (% per year)", 10, 40, 25) / 100
p.order_cost = st.sidebar.number_input("Ordering cost per order (€)", 10, 500, 75, step=5)
p.uniform_sl = st.sidebar.slider("Uniform policy: service level (%)", 80.0, 99.5, 95.0, 0.5) / 100
with st.sidebar.expander("Segmented service levels (%)"):
    for seg, default in list(p.segment_sl.items()):
        p.segment_sl[seg] = st.slider(seg, 80.0, 99.5, default * 100, 0.5, key=seg) / 100
st.sidebar.caption("Demo data is synthetic: 300 spare parts, 52 weeks of demand, no company data.")

master, demand, stock = load_data()
df = an.add_excess(an.add_projection(compute(master, demand, stock, p)), demand)
k = kpis(df)
ik = an.inventory_kpis(df)


@st.cache_data
def _forecast(demand, master):
    return an.forecast_backtest(demand, master)


@st.cache_data
def _po_history():
    return an.load_po_history()


fc, fc_detail = _forecast(demand, master)
po_hist = _po_history()
perf = an.supplier_performance(po_hist, df)

st.title("📦 AI Inventory Planning Copilot")
st.caption("ABC/XYZ segmentation · safety stock · reorder points · stock-out risk · excess stock · "
           "supplier performance · forecast accuracy · AI assistant for planners")

(tab_overview, tab_alerts, tab_risk, tab_part, tab_excess, tab_supplier, tab_fc, tab_ai) = st.tabs(
    ["📊 Overview", "🚨 Reorder alerts", "⏳ Stock-out risk", "🔍 Part explorer", "💰 Excess & slow movers",
     "🚚 Supplier performance", "🎯 Forecast accuracy", "🤖 AI Copilot"])

# ---------- Overview ----------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg stock value — uniform", f"{k['uniform_avg_stock_eur']:,.0f} €")
    c2.metric("Avg stock value — segmented", f"{k['segmented_avg_stock_eur']:,.0f} €",
              f"-{k['saving_eur']:,.0f} € ({k['saving_pct']}%)", delta_color="inverse")
    c3.metric("Weighted service level — uniform", f"{k['uniform_weighted_service_level_pct']}%")
    c4.metric("Weighted service level — segmented", f"{k['segmented_weighted_service_level_pct']}%")

    st.markdown("**Inventory health today**")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Stock value on hand", f"{ik['stock_value_on_hand_eur']:,.0f} €")
    d2.metric("Inventory turns (actual / target)",
              f"{ik['inventory_turns_actual']} / {ik['inventory_turns_target_segmented_policy']}",
              help="Annual consumption value ÷ stock value. Target = with the segmented policy's average stock.")
    d3.metric("Days of supply", f"{ik['days_of_supply_actual']:.0f} days",
              help="Stock value ÷ average daily consumption value")
    d4.metric("Excess stock value", f"{ik['excess_value_eur']:,.0f} €",
              help="Stock above the planned maximum (reorder point + order quantity)")

    st.subheader("Pareto: which parts make up the annual consumption value? (ABC)")
    par = df.sort_values("annual_value_eur", ascending=False).reset_index(drop=True)
    par["rank"] = par.index + 1
    par["cum_pct"] = par["annual_value_eur"].cumsum() / par["annual_value_eur"].sum()
    base = alt.Chart(par).encode(x=alt.X("rank:Q", title="Parts ranked by annual value"))
    bars = base.mark_bar().encode(
        y=alt.Y("annual_value_eur:Q", title="Annual value (€)"),
        color=alt.Color("abc:N", title="ABC", scale=alt.Scale(domain=["A", "B", "C"],
                                                               range=["#185FA5", "#5DCAA5", "#B4B2A9"])),
        tooltip=["part", "description", "abc", alt.Tooltip("annual_value_eur:Q", format=",.0f")])
    line = base.mark_line(color="#D85A30").encode(
        y=alt.Y("cum_pct:Q", title="Cumulative share", axis=alt.Axis(format="%")))
    st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent"), width="stretch")
    n_a = int((par["abc"] == "A").sum())
    st.caption(f"{n_a} A-parts ({n_a / len(par):.0%} of parts) make up 80% of the annual consumption value.")

    left, right = st.columns(2)
    with left:
        st.subheader("ABC/XYZ matrix (number of parts)")
        st.dataframe(pd.crosstab(df["abc"], df["xyz"]), width="stretch")
        st.subheader("Average stock value by ABC class (€)")
        by = df.groupby("abc")[["uniform_avg_stock_eur", "segment_avg_stock_eur"]].sum()
        by.columns = ["Uniform", "Segmented"]
        st.bar_chart(by.reset_index().melt("abc", var_name="Policy", value_name="EUR"),
                     x="abc", y="EUR", color="Policy", stack=False, x_label="ABC class", y_label="€")
    with right:
        st.subheader("Each part in the ABC/XYZ grid")
        sc = alt.Chart(df).mark_circle(size=40, opacity=0.6).encode(
            x=alt.X("annual_value_eur:Q", title="Annual value (€, log scale)", scale=alt.Scale(type="log")),
            y=alt.Y("cv:Q", title="Demand variability (CV)"),
            color=alt.Color("segment:N", title="Segment"),
            tooltip=["part", "description", "segment", alt.Tooltip("cv:Q", format=".2f"),
                     alt.Tooltip("annual_value_eur:Q", format=",.0f")])
        rules = alt.Chart(pd.DataFrame({"y": [0.5, 1.0]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y")
        st.altair_chart(sc + rules, width="stretch")
        st.subheader("Stock status today")
        st.bar_chart(df["status"].value_counts().sort_index())

# ---------- Alerts ----------
with tab_alerts:
    st.subheader("Parts that need action")
    alert = df[df["status"].str.startswith(("1", "2", "3"))].sort_values(["status", "weeks_to_stockout"])
    sup = st.selectbox("Filter by supplier", ["All"] + sorted(df["supplier"].unique()))
    if sup != "All":
        alert = alert[alert["supplier"] == sup]
    st.dataframe(alert[["part", "description", "supplier", "segment", "status", "on_hand_units",
                        "segment_ss", "reorder_point", "open_po_units", "po_due_in_weeks",
                        "weeks_to_stockout", "suggested_order_units"]].rename(columns={"segment_ss": "safety_stock"}),
                 width="stretch", hide_index=True)
    st.caption("weeks_to_stockout = week in which projected stock (stock + open PO − average demand) drops below zero.")
    hidden = df[df["status"].str.startswith(("3", "4"))
                & (df["open_po_units"] > 0) & (df["weeks_to_stockout"] < df["po_due_in_weeks"])]
    if len(hidden):
        st.subheader(f"Hidden risk: {len(hidden)} parts look OK but run out before their open PO arrives")
        st.caption("Their stock plus open PO is above the reorder point, so the status rule says OK. "
                   "But the PO comes too late. Action: ask the supplier to deliver earlier.")
        st.dataframe(hidden[["part", "description", "supplier", "status", "on_hand_units", "avg_weekly",
                             "open_po_units", "po_due_in_weeks", "weeks_to_stockout"]],
                     width="stretch", hide_index=True)
    st.caption("Tip: ask the AI Copilot to draft a German reminder email for a supplier in this list.")

# ---------- Stock-out risk ----------
with tab_risk:
    st.subheader("When will parts run out? (projection without new orders)")
    horizon = st.slider("Show parts that run out within … weeks", 1, 26, 8)
    risk = df[df["weeks_to_stockout"] <= horizon].copy()
    r1, r2, r3 = st.columns(3)
    r1.metric(f"Parts running out within {horizon} weeks", len(risk))
    r2.metric("…of these, A-parts", int((risk["abc"] == "A").sum()))
    r3.metric("Run out before a new order can arrive", int(risk["stockout_before_order_arrives"].sum()),
              help="Stock-out week is earlier than the lead time: only expediting or alternatives help.")
    cnt = risk.groupby("weeks_to_stockout").size().rename("parts").reset_index()
    st.altair_chart(alt.Chart(cnt).mark_bar().encode(
        x=alt.X("weeks_to_stockout:O", title="Week in which stock runs out"),
        y=alt.Y("parts:Q", title="Number of parts")), width="stretch")
    risk["action"] = np.where(risk["stockout_before_order_arrives"], "Expedite / alternative source",
                              "Order now")
    st.dataframe(risk.sort_values(["weeks_to_stockout", "annual_value_eur"], ascending=[True, False])[
        ["part", "description", "supplier", "segment", "status", "on_hand_units", "avg_weekly",
         "open_po_units", "po_due_in_weeks", "lead_time_weeks", "weeks_to_stockout", "action"]],
        width="stretch", hide_index=True)
    st.caption("Uses average weekly demand. Parts with irregular (Z) demand can run out earlier or later.")

# ---------- Part explorer ----------
with tab_part:
    part = st.selectbox("Part", df["part"] + " — " + df["description"])
    r = df[df["part"] == part.split(" — ")[0]].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Segment", r.segment)
    b.metric("Safety stock (units)", f"{r.segment_ss:,.0f}")
    c.metric("Reorder point (units)", f"{r.reorder_point:,.0f}")
    d.metric("Runs out in week", "—" if np.isnan(r.weeks_to_stockout) else f"{r.weeks_to_stockout:.0f}")

    st.subheader("Projected stock (next 26 weeks)")
    with_order = st.toggle("Include the suggested order (arrives after the lead time)",
                           value=bool(r.suggested_order_units > 0))
    proj = an.project_part(r, with_order=with_order)
    lines = proj.melt("week", var_name="series", value_name="units")
    lines["series"] = lines["series"].map({"projected_stock": "Projected stock", "safety_stock": "Safety stock",
                                           "reorder_point": "Reorder point"})
    ch = alt.Chart(lines).mark_line(interpolate="step-after").encode(
        x=alt.X("week:Q", title="Weeks from today"), y=alt.Y("units:Q", title="Units"),
        color=alt.Color("series:N", title=None, scale=alt.Scale(
            domain=["Projected stock", "Reorder point", "Safety stock"], range=["#185FA5", "#EF9F27", "#E24B4A"])),
        strokeDash=alt.condition(alt.datum.series == "Projected stock", alt.value([1, 0]), alt.value([5, 4])))
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray").encode(y="y")
    st.altair_chart(ch + zero, width="stretch")
    st.caption("Like MD04 as a picture: stock today, plus open PO in its due week, minus average weekly demand.")

    st.markdown(
        f"""
**How the numbers are calculated**
- Average weekly demand = **{r.avg_weekly:,.1f}**, standard deviation = **{r.std_weekly:,.1f}**, CV = **{r.cv:.2f}** → XYZ class **{r.xyz}**
- Annual value = {r.annual_demand:,.0f} units × {r.unit_cost_eur:,.2f} € = **{r.annual_value_eur:,.0f} €** → ABC class **{r.abc}**
- Service level for {r.segment} = **{r.segment_sl:.1%}**
- Safety stock = z × σ × √LT = NORMSINV({r.segment_sl:.3f}) × {r.std_weekly:,.1f} × √{r.lead_time_weeks} = **{r.segment_ss:,.0f}** units
- Reorder point = {r.avg_weekly:,.1f} × {r.lead_time_weeks} + {r.segment_ss:,.0f} = **{r.reorder_point:,.0f}** units
- EOQ = √(2 × {r.annual_demand:,.0f} × {p.order_cost:.0f} ÷ ({r.unit_cost_eur:,.2f} × {p.holding_rate:.2f})) = **{r.eoq:,.0f}** units
- Today: on hand **{r.on_hand_units:,.0f}**, open PO **{r.open_po_units:,.0f}** → **{r.status[2:]}**
"""
    )
    st.subheader("Weekly demand and forecast check (last 13 weeks)")
    hist = demand[demand["part"] == r.part].drop(columns="part").T
    hist.columns = ["Weekly demand"]
    st.line_chart(hist)
    fr = fc[fc["part"] == r.part].iloc[0]
    fd = pd.DataFrame({"week": fc_detail["weeks"], "Actual": fc_detail["actual"].loc[r.part].values,
                       "Moving avg 8 wks": fc_detail["ma8"].loc[r.part].values,
                       "Exp. smoothing": fc_detail["ses"].loc[r.part].values})
    st.line_chart(fd.set_index("week"))
    st.caption(f"Forecast error (WMAPE): moving average {fr.ma8_wmape:.0%}, exponential smoothing "
               f"{fr.ses_wmape:.0%} → better method: {fr.best_method}.")

# ---------- Excess & slow movers ----------
with tab_excess:
    st.subheader("Capital tied up in excess and slow-moving stock")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Excess stock value", f"{ik['excess_value_eur']:,.0f} €")
    e2.metric("Parts with excess", ik["parts_with_excess"])
    e3.metric("Slow / non-moving parts", ik["slow_or_non_moving_parts"])
    e4.metric("Slow / non-moving stock value", f"{ik['slow_or_non_moving_value_eur']:,.0f} €")
    st.caption("Excess = stock on hand above the planned maximum (reorder point + order quantity). "
               "Slow-moving = no demand in the last 13 weeks; non-moving = none in 26 weeks.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Stock value by weeks of cover**")
        bucket = df.groupby("cover_bucket", observed=False)["stock_value_eur"].sum().reset_index()
        st.altair_chart(alt.Chart(bucket).mark_bar().encode(
            x=alt.X("cover_bucket:N", title="Weeks of cover", sort=None),
            y=alt.Y("stock_value_eur:Q", title="Stock value (€)"),
            color=alt.Color("cover_bucket:N", legend=None, scale=alt.Scale(
                range=["#E24B4A", "#EF9F27", "#5DCAA5", "#1D9E75", "#7F77DD", "#534AB7"]))), width="stretch")
    with right:
        st.markdown("**Top 15 parts by excess value**")
        top = df[df["excess_units"] > 0].nlargest(15, "excess_value_eur")
        st.altair_chart(alt.Chart(top).mark_bar(color="#534AB7").encode(
            x=alt.X("excess_value_eur:Q", title="Excess value (€)"),
            y=alt.Y("part:N", sort="-x", title=None),
            tooltip=["part", "description", "supplier", alt.Tooltip("excess_units:Q", format=",.0f"),
                     alt.Tooltip("weeks_of_cover:Q", format=".1f")]), width="stretch")
    st.markdown("**Excess and slow-moving parts**")
    ex = df[(df["excess_units"] > 0) | (df["movement"] != "Normal")].sort_values("excess_value_eur",
                                                                                 ascending=False)
    st.dataframe(ex[["part", "description", "category", "supplier", "segment", "on_hand_units",
                     "max_stock_units", "excess_units", "excess_value_eur", "weeks_of_cover", "movement",
                     "open_po_units"]], width="stretch", hide_index=True)
    st.caption("Typical actions: cancel or push out open POs, stop reordering, use up stock, return to supplier.")

# ---------- Supplier performance ----------
with tab_supplier:
    st.subheader("On-time delivery (OTD) by supplier — last 12 months")
    s1, s2, s3 = st.columns(3)
    s1.metric("Overall OTD", f"{po_hist['on_time'].mean():.1%}")
    s2.metric("PO lines delivered", f"{len(po_hist):,}")
    s3.metric("Suppliers below 85%", int((perf["on_time_pct"] < 85).sum()))
    bars = alt.Chart(perf).mark_bar().encode(
        x=alt.X("on_time_pct:Q", title="On-time delivery %", scale=alt.Scale(domain=[50, 100])),
        y=alt.Y("supplier:N", sort="x", title=None),
        color=alt.Color("rating:N", title="Rating", scale=alt.Scale(
            domain=["A (≥95%)", "B (85-95%)", "C (<85%)"], range=["#1D9E75", "#EF9F27", "#E24B4A"])),
        tooltip=["supplier", "po_lines", "on_time_pct", alt.Tooltip("avg_days_late_when_late:Q", format=".1f")])
    target = alt.Chart(pd.DataFrame({"x": [95]})).mark_rule(strokeDash=[4, 4]).encode(x="x")
    st.altair_chart(bars + target, width="stretch")
    st.caption("Dashed line = 95% target. On time = delivered on or before the confirmed date.")
    st.dataframe(perf.rename(columns={"avg_days_late_when_late": "avg_days_late (late lines)"}).round(1),
                 width="stretch", hide_index=True)
    pick = st.selectbox("Monthly trend for supplier", perf["supplier"].tolist())
    t = po_hist[po_hist["supplier"] == pick].copy()
    t["month"] = t["promised_date"].dt.to_period("M").dt.to_timestamp()
    trend = t.groupby("month")["on_time"].mean().rename("OTD").reset_index()
    st.altair_chart(alt.Chart(trend).mark_line(point=True).encode(
        x=alt.X("month:T", title=None), y=alt.Y("OTD:Q", axis=alt.Axis(format="%"),
                                                scale=alt.Scale(domain=[0, 1]))), width="stretch")
    st.caption("Purchase order history is synthetic demo data.")

# ---------- Forecast accuracy ----------
with tab_fc:
    st.subheader("How good would a simple forecast be? (back-test on the last 13 weeks)")
    st.caption("Each week is forecast with the data known before it: moving average of 8 weeks, and "
               "exponential smoothing (α = 0.2). WMAPE = total absolute error ÷ total demand (lower is better). "
               "Bias > 0 = forecast too high, < 0 = too low. Value-weighted, so expensive parts count more.")
    fx = an.forecast_summary(fc, df, "xyz")
    fa = an.forecast_summary(fc, df, "abc")
    tot = fc.merge(df[["part", "unit_cost_eur"]], on="part")
    w_err = (tot[["ma8_abs_err", "ses_abs_err"]].mul(tot["unit_cost_eur"], axis=0).sum()
             / (tot["actual_units"] * tot["unit_cost_eur"]).sum())
    f1, f2, f3 = st.columns(3)
    f1.metric("WMAPE moving average", f"{w_err['ma8_abs_err']:.0%}")
    f2.metric("WMAPE exponential smoothing", f"{w_err['ses_abs_err']:.0%}")
    f3.metric("Forecast accuracy (1 − WMAPE)", f"{1 - w_err.min():.0%}")
    left, right = st.columns(2)
    with left:
        st.markdown("**By XYZ class**")
        st.dataframe(fx.style.format("{:.0%}"), width="stretch")
    with right:
        st.markdown("**By ABC class**")
        st.dataframe(fa.style.format("{:.0%}"), width="stretch")
    st.caption("As expected, X-parts (stable demand) forecast well and Z-parts (irregular) badly. "
               "This is why Z-parts need more safety stock relative to demand.")
    xs = fc.merge(df[["part", "description", "cv", "xyz", "abc", "annual_value_eur"]], on="part")
    st.altair_chart(alt.Chart(xs.dropna(subset=["best_wmape"])).mark_circle(size=40, opacity=0.6).encode(
        x=alt.X("cv:Q", title="Demand variability (CV)"),
        y=alt.Y("best_wmape:Q", title="Forecast error (WMAPE)", axis=alt.Axis(format="%")),
        color=alt.Color("xyz:N", title="XYZ"),
        tooltip=["part", "description", alt.Tooltip("best_wmape:Q", format=".0%"), "best_method"]),
        width="stretch")
    st.markdown("**A-parts with the highest forecast error** (most important to review with sales / production)")
    st.dataframe(xs[xs["abc"] == "A"].nlargest(15, "best_wmape")[
        ["part", "description", "xyz", "cv", "ma8_wmape", "ses_wmape", "ses_bias", "best_method"]].round(2),
        width="stretch", hide_index=True)

# ---------- AI Copilot ----------
with tab_ai:
    st.subheader("Ask the planning copilot")
    st.caption("The AI answers using tool calls on the planning data above, so its numbers are traceable. "
               "Supplier emails are drafts for you to review, nothing is sent.")
    if not API_KEY:
        st.info("To activate the AI Copilot, add your free GEMINI_API_KEY in Streamlit → Settings → Secrets "
                "(or as an environment variable when running locally). The other tabs work without it.")
    elif APP_PASSWORD and not st.session_state.get("unlocked"):
        pw = st.text_input("The AI chat has a limited free daily quota. Enter the demo password to use it:", type="password")
        if pw:
            if pw == APP_PASSWORD:
                st.session_state.unlocked = True
                st.rerun()
            else:
                st.error("Wrong password.")
        st.caption("No password? The Overview, Reorder alerts and Part explorer tabs are open to everyone.")
    else:
        copilot.CONTEXT.update(fc=fc, perf=perf, po=po_hist)
        if "model" not in st.session_state:
            st.session_state.model = copilot.model_candidates(MODEL_PREF)
        if "history" not in st.session_state:
            st.session_state.history, st.session_state.display = [], []

        examples = ["Which parts are critical right now, and what should I order?",
                    "Explain why MP-10001 has this safety stock.",
                    "What happens to MP-10001 if I raise its service level to 98%?",
                    "Draft a German reminder email to Supplier 03 for their late and urgent parts."]
        cols = st.columns(len(examples))
        clicked = None
        for col, ex in zip(cols, examples):
            if col.button(ex, width="stretch"):
                clicked = ex

        for role, text, calls in st.session_state.display:
            with st.chat_message(role):
                st.markdown(text)
                if calls:
                    with st.expander(f"Data used ({len(calls)} tool calls)"):
                        for n, args in calls:
                            st.code(f"{n}({args})")

        prompt = st.chat_input("e.g. Which suppliers cause the most critical parts?") or clicked
        if prompt:
            st.session_state.display.append(("user", prompt, []))
            turn_start = len(st.session_state.history)
            st.session_state.history.append({"role": "user", "parts": [{"text": prompt}]})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Checking the planning data…"):
                    try:
                        answer, calls = copilot.chat(API_KEY, st.session_state.model, st.session_state.history, df, p,
                                                     lambda d: {**kpis(d), **an.inventory_kpis(d)})
                    except Exception as e:
                        answer, calls = f"⚠️ {e}", []
                        del st.session_state.history[turn_start:]  # drop the failed turn
                st.markdown(answer)
                if calls:
                    with st.expander(f"Data used ({len(calls)} tool calls)"):
                        for n, args in calls:
                            st.code(f"{n}({args})")
            st.session_state.display.append(("assistant", answer, calls))

        if st.session_state.get("display") and st.button("Clear chat"):
            st.session_state.history, st.session_state.display = [], []
            st.rerun()
