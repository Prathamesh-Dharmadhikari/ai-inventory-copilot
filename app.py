"""AI Inventory Planning Copilot — Streamlit app.

Run locally:  streamlit run app.py
Deploy free:  share.streamlit.io (connect this GitHub repo, add GEMINI_API_KEY under Secrets — free key from aistudio.google.com)
"""
import os

import pandas as pd
import streamlit as st

from planning import Params, compute, kpis, load_data
import copilot

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
df = compute(master, demand, stock, p)
k = kpis(df)

st.title("📦 AI Inventory Planning Copilot")
st.caption("ABC/XYZ segmentation · safety stock · reorder points · reorder alerts · AI assistant for planners")

tab_overview, tab_alerts, tab_part, tab_ai = st.tabs(
    ["📊 Overview", "🚨 Reorder alerts", "🔍 Part explorer", "🤖 AI Copilot"])

# ---------- Overview ----------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg stock value — uniform", f"{k['uniform_avg_stock_eur']:,.0f} €")
    c2.metric("Avg stock value — segmented", f"{k['segmented_avg_stock_eur']:,.0f} €",
              f"-{k['saving_eur']:,.0f} € ({k['saving_pct']}%)", delta_color="inverse")
    c3.metric("Weighted service level — uniform", f"{k['uniform_weighted_service_level_pct']}%")
    c4.metric("Weighted service level — segmented", f"{k['segmented_weighted_service_level_pct']}%")

    left, right = st.columns(2)
    with left:
        st.subheader("ABC/XYZ matrix (number of parts)")
        st.dataframe(pd.crosstab(df["abc"], df["xyz"]), width="stretch")
        st.subheader("Share of annual value")
        share = pd.crosstab(df["abc"], df["xyz"], values=df["annual_value_eur"], aggfunc="sum",
                            normalize="all").fillna(0)
        st.dataframe(share.style.format("{:.1%}"), width="stretch")
    with right:
        st.subheader("Average stock value by ABC class (€)")
        by = df.groupby("abc")[["uniform_avg_stock_eur", "segment_avg_stock_eur"]].sum()
        by.columns = ["Uniform", "Segmented"]
        st.bar_chart(by.reset_index().melt("abc", var_name="Policy", value_name="EUR"),
                     x="abc", y="EUR", color="Policy", stack=False, x_label="ABC class", y_label="€")
        st.subheader("Stock status today")
        st.bar_chart(df["status"].value_counts().sort_index())

# ---------- Alerts ----------
with tab_alerts:
    st.subheader("Parts that need action")
    alert = df[df["status"].str.startswith(("1", "2", "3"))].sort_values(["status", "annual_value_eur"],
                                                                         ascending=[True, False])
    sup = st.selectbox("Filter by supplier", ["All"] + sorted(df["supplier"].unique()))
    if sup != "All":
        alert = alert[alert["supplier"] == sup]
    st.dataframe(alert[["part", "description", "supplier", "segment", "status", "on_hand_units",
                        "segment_ss", "reorder_point", "open_po_units", "po_due_in_weeks",
                        "suggested_order_units"]].rename(columns={"segment_ss": "safety_stock"}),
                 width="stretch", hide_index=True)
    st.caption("Tip: ask the AI Copilot to draft a German reminder email for a supplier in this list.")

# ---------- Part explorer ----------
with tab_part:
    part = st.selectbox("Part", df["part"] + " — " + df["description"])
    r = df[df["part"] == part.split(" — ")[0]].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Segment", r.segment)
    b.metric("Safety stock (units)", f"{r.segment_ss:,.0f}")
    c.metric("Reorder point (units)", f"{r.reorder_point:,.0f}")
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
    hist = demand[demand["part"] == r.part].drop(columns="part").T
    hist.columns = ["Weekly demand"]
    st.line_chart(hist)

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
                        answer, calls = copilot.chat(API_KEY, st.session_state.model, st.session_state.history, df, p, kpis)
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
