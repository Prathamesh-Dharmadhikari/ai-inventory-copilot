"""Planner analytics on top of the planning engine (planning.py).

1. Stock projection: when will a part reach its reorder point, safety stock and zero?
2. Excess and slow-moving stock: capital tied up above the planned maximum
3. Inventory KPIs: stock value, inventory turns, days of supply
4. Forecast accuracy: back-test of two simple forecasts on the last 13 weeks (WMAPE, bias)
5. Supplier performance: on-time delivery from the purchase order history
All data is synthetic.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
HORIZON = 26          # weeks shown in the stock projection
TEST_WEEKS = 13       # forecast back-test: the last quarter


# ---------------------------------------------------------------- 1. stock projection
def project_part(r, weeks: int = HORIZON, with_order: bool = False) -> pd.DataFrame:
    """Week-by-week projected stock = stock today + open PO (in its due week) - average weekly demand.
    with_order=True adds the suggested order, arriving after the lead time."""
    stock = float(r.on_hand_units)
    po_week = max(int(r.po_due_in_weeks), 1) if r.open_po_units > 0 else None
    order_week = int(r.lead_time_weeks) if with_order and r.suggested_order_units > 0 else None
    rows = []
    for w in range(0, weeks + 1):
        if w > 0:
            stock -= r.avg_weekly
            if po_week == w:
                stock += r.open_po_units
            if order_week == w:
                stock += r.suggested_order_units
        rows.append({"week": w, "projected_stock": round(stock, 1),
                     "safety_stock": r.segment_ss, "reorder_point": r.reorder_point})
    return pd.DataFrame(rows)


def _first_week(proj: pd.DataFrame, limit) -> float:
    hit = proj.loc[proj["projected_stock"] < limit, "week"]
    return float(hit.iloc[0]) if len(hit) else np.nan


def add_projection(df: pd.DataFrame) -> pd.DataFrame:
    """Adds weeks until stock-out / below safety stock (no new orders), within the 52-week horizon."""
    out = df.copy()
    so, bss = [], []
    for r in out.itertuples():
        proj = project_part(r, weeks=52)
        so.append(_first_week(proj, 0))
        bss.append(_first_week(proj, r.segment_ss))
    out["weeks_to_stockout"] = so          # NaN = no stock-out within 52 weeks
    out["weeks_to_below_ss"] = bss
    out["stockout_before_order_arrives"] = out["weeks_to_stockout"] < out["lead_time_weeks"]
    return out


# ---------------------------------------------------------------- 2. excess and slow movers
def add_excess(df: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["max_stock_units"] = out["reorder_point"] + out["eoq"]           # planned maximum
    out["excess_units"] = np.maximum(out["on_hand_units"] - out["max_stock_units"], 0)
    out["excess_value_eur"] = out["excess_units"] * out["unit_cost_eur"]
    out["stock_value_eur"] = out["on_hand_units"] * out["unit_cost_eur"]

    hist = demand.set_index("part").loc[out["part"]].to_numpy(dtype=float)
    last13 = hist[:, -13:].sum(axis=1)
    out["demand_last_13w"] = last13
    out["demand_last_26w"] = hist[:, -26:].sum(axis=1)
    out["movement"] = np.where(out["demand_last_26w"] == 0, "Non-moving (no demand 26 wks)",
                       np.where(last13 == 0, "Slow-moving (no demand 13 wks)",
                       np.where(last13 < 0.5 * hist[:, -26:-13].sum(axis=1), "Falling demand (-50%+)",
                                "Normal")))
    woc = out["weeks_of_cover"].replace(np.inf, 999)
    out["cover_bucket"] = pd.cut(woc, [-0.1, 2, 4, 8, 13, 26, 1e9],
                                 labels=["0-2 wks", "2-4 wks", "4-8 wks", "8-13 wks", "13-26 wks", ">26 wks"])
    return out


# ---------------------------------------------------------------- 3. inventory KPIs
def inventory_kpis(dfx: pd.DataFrame) -> dict:
    value = dfx["stock_value_eur"].sum()
    annual = dfx["annual_value_eur"].sum()
    target = dfx["segment_avg_stock_eur"].sum()
    return {
        "stock_value_on_hand_eur": int(round(value)),
        "inventory_turns_actual": round(annual / value, 1) if value else None,
        "inventory_turns_target_segmented_policy": round(annual / target, 1) if target else None,
        "days_of_supply_actual": round(value / (annual / 365), 1) if annual else None,
        "excess_value_eur": int(round(dfx["excess_value_eur"].sum())),
        "parts_with_excess": int((dfx["excess_units"] > 0).sum()),
        "slow_or_non_moving_parts": int(dfx["movement"].str.startswith(("Slow", "Non")).sum()),
        "slow_or_non_moving_value_eur": int(round(
            dfx.loc[dfx["movement"].str.startswith(("Slow", "Non")), "stock_value_eur"].sum())),
        "parts_stockout_within_4_weeks": int((dfx["weeks_to_stockout"] <= 4).sum()),
        "parts_stockout_before_new_order_can_arrive": int(dfx["stockout_before_order_arrives"].sum()),
    }


# ---------------------------------------------------------------- 4. forecast accuracy
def forecast_backtest(demand: pd.DataFrame, master: pd.DataFrame, test_weeks: int = TEST_WEEKS):
    """One-step-ahead forecasts for the last `test_weeks` weeks:
    - Moving average of the previous 8 weeks
    - Exponential smoothing (alpha 0.2)
    WMAPE = sum |forecast - actual| / sum actual;  bias = sum (forecast - actual) / sum actual."""
    hist = demand.set_index("part")
    parts = hist.index
    y = hist.to_numpy(dtype=float)
    n = y.shape[1]
    start = n - test_weeks
    ma = np.stack([y[:, t - 8:t].mean(axis=1) for t in range(start, n)], axis=1)
    level = y[:, :8].mean(axis=1)
    ses_cols = []
    for t in range(8, n):
        if t >= start:
            ses_cols.append(level.copy())
        level = 0.2 * y[:, t] + 0.8 * level
    ses = np.stack(ses_cols, axis=1)
    actual = y[:, start:]

    def metrics(f):
        abs_err, err, act = np.abs(f - actual).sum(1), (f - actual).sum(1), actual.sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return abs_err, err, act, np.where(act > 0, abs_err / act, np.nan), np.where(act > 0, err / act, np.nan)

    ma_abs, ma_err, act, ma_wmape, ma_bias = metrics(ma)
    se_abs, se_err, _, se_wmape, se_bias = metrics(ses)
    res = pd.DataFrame({"part": parts, "actual_units": act,
                        "ma8_wmape": ma_wmape, "ma8_bias": ma_bias, "ses_wmape": se_wmape, "ses_bias": se_bias,
                        "ma8_abs_err": ma_abs, "ses_abs_err": se_abs, "ma8_err": ma_err, "ses_err": se_err})
    res["best_method"] = np.where(res["ses_wmape"] < res["ma8_wmape"], "Exp. smoothing", "Moving avg 8 wks")
    res["best_wmape"] = res[["ma8_wmape", "ses_wmape"]].min(axis=1)
    detail = {"weeks": [f"W{i:02d}" for i in range(start + 1, n + 1)],
              "actual": pd.DataFrame(actual, index=parts), "ma8": pd.DataFrame(ma, index=parts),
              "ses": pd.DataFrame(ses, index=parts)}
    return res, detail


def forecast_summary(fc: pd.DataFrame, df: pd.DataFrame, by: str = "xyz") -> pd.DataFrame:
    """Value-weighted WMAPE and bias per group (so expensive parts count more)."""
    x = fc.merge(df[["part", "unit_cost_eur", by]], on="part")
    g = x.assign(a=x.actual_units * x.unit_cost_eur,
                 ma_abs=x.ma8_abs_err * x.unit_cost_eur, se_abs=x.ses_abs_err * x.unit_cost_eur,
                 ma_e=x.ma8_err * x.unit_cost_eur, se_e=x.ses_err * x.unit_cost_eur).groupby(by)[
        ["a", "ma_abs", "se_abs", "ma_e", "se_e"]].sum()
    return pd.DataFrame({
        "WMAPE moving avg": g.ma_abs / g.a, "WMAPE exp. smoothing": g.se_abs / g.a,
        "Bias moving avg": g.ma_e / g.a, "Bias exp. smoothing": g.se_e / g.a,
    })


# ---------------------------------------------------------------- 5. supplier performance
def load_po_history() -> pd.DataFrame:
    po = pd.read_csv(DATA / "po_history.csv", parse_dates=["promised_date", "delivered_date"])
    po["days_late"] = (po["delivered_date"] - po["promised_date"]).dt.days
    po["on_time"] = po["days_late"] <= 0
    return po


def supplier_performance(po: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    perf = po.groupby("supplier").agg(
        po_lines=("po_line", "count"),
        on_time_pct=("on_time", "mean"),
        avg_days_late_when_late=("days_late", lambda d: d[d > 0].mean() if (d > 0).any() else 0.0),
        max_days_late=("days_late", "max"),
    )
    perf["on_time_pct"] = (perf["on_time_pct"] * 100).round(1)
    cur = df.groupby("supplier").agg(
        open_pos_confirmed_late=("po_confirmed_late", "sum"),
        critical_or_reorder_parts=("status", lambda s: s.str.startswith(("1", "2")).sum()),
        annual_value_eur=("annual_value_eur", "sum"),
    )
    out = perf.join(cur).sort_values("on_time_pct")
    out["rating"] = np.where(out["on_time_pct"] >= 95, "A (≥95%)",
                     np.where(out["on_time_pct"] >= 85, "B (85-95%)", "C (<85%)"))
    return out.reset_index()
