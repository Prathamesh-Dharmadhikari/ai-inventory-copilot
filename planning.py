"""Planning engine: the same logic as the Excel model (Inventory_Policy_Optimizer.xlsx).

ABC/XYZ segmentation -> safety stock -> reorder point -> EOQ -> stock value,
plus a stock-status check against the current stock snapshot.
All data is synthetic.
"""
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"


@dataclass
class Params:
    holding_rate: float = 0.25        # per year, share of unit cost
    order_cost: float = 75.0          # € per purchase order
    abc_a: float = 0.80               # cumulative value share for A
    abc_b: float = 0.95               # cumulative value share for B
    xyz_x: float = 0.5                # max CV for X
    xyz_y: float = 1.0                # max CV for Y
    uniform_sl: float = 0.95          # baseline: one service level for all parts
    segment_sl: dict = field(default_factory=lambda: {
        "AX": 0.95, "AY": 0.92, "AZ": 0.88,
        "BX": 0.97, "BY": 0.95, "BZ": 0.92,
        "CX": 0.99, "CY": 0.99, "CZ": 0.98,
    })


def z(sl: float) -> float:
    return NormalDist().inv_cdf(sl)


def load_data():
    master = pd.read_csv(DATA / "master_data.csv")
    demand = pd.read_csv(DATA / "demand_history.csv")
    stock = pd.read_csv(DATA / "stock_snapshot.csv")
    return master, demand, stock


def compute(master: pd.DataFrame, demand: pd.DataFrame, stock: pd.DataFrame, p: Params) -> pd.DataFrame:
    df = master.copy()
    weeks = demand.set_index("part").loc[df["part"]].to_numpy(dtype=float)
    df["annual_demand"] = weeks.sum(axis=1)
    df["avg_weekly"] = weeks.mean(axis=1)
    df["std_weekly"] = weeks.std(axis=1, ddof=1)
    df["cv"] = np.where(df["avg_weekly"] > 0, df["std_weekly"] / df["avg_weekly"], 0)
    df["annual_value_eur"] = df["annual_demand"] * df["unit_cost_eur"]

    total = df["annual_value_eur"].sum()
    v = df["annual_value_eur"].to_numpy()
    df["cum_value_share"] = [v[v >= x].sum() / total for x in v]   # same as Excel SUMIF(">=")
    df["abc"] = np.where(df["cum_value_share"] <= p.abc_a, "A",
                         np.where(df["cum_value_share"] <= p.abc_b, "B", "C"))
    df["xyz"] = np.where(df["cv"] <= p.xyz_x, "X", np.where(df["cv"] <= p.xyz_y, "Y", "Z"))
    df["segment"] = df["abc"] + df["xyz"]

    sqrt_lt = np.sqrt(df["lead_time_weeks"])
    df["uniform_sl"] = p.uniform_sl
    df["uniform_ss"] = np.ceil(z(p.uniform_sl) * df["std_weekly"] * sqrt_lt)
    df["segment_sl"] = df["segment"].map(p.segment_sl)
    df["segment_ss"] = np.ceil(df["segment_sl"].map(z) * df["std_weekly"] * sqrt_lt)
    df["reorder_point"] = np.ceil(df["avg_weekly"] * df["lead_time_weeks"]) + df["segment_ss"]
    df["eoq"] = np.where(
        df["annual_demand"] > 0,
        np.ceil(np.sqrt(2 * df["annual_demand"] * p.order_cost / (df["unit_cost_eur"] * p.holding_rate))),
        0,
    )
    df["uniform_avg_stock_eur"] = (df["eoq"] / 2 + df["uniform_ss"]) * df["unit_cost_eur"]
    df["segment_avg_stock_eur"] = (df["eoq"] / 2 + df["segment_ss"]) * df["unit_cost_eur"]

    # --- current stock situation ---
    df = df.merge(stock, on="part", how="left")
    df["available_units"] = df["on_hand_units"] + df["open_po_units"]
    df["weeks_of_cover"] = np.where(df["avg_weekly"] > 0, df["on_hand_units"] / df["avg_weekly"], np.inf)

    def status(r):
        if r.on_hand_units < r.segment_ss:
            return "1 Critical: below safety stock"
        if r.available_units < r.reorder_point:
            return "2 Reorder now: below reorder point, no sufficient PO"
        if r.on_hand_units < r.reorder_point and r.po_confirmed_late == 1:
            return "3 Watch: open PO confirmed late"
        if r.on_hand_units > r.reorder_point + 2 * r.eoq:
            return "5 Overstock"
        return "4 OK"
    df["status"] = df.apply(status, axis=1)
    df["suggested_order_units"] = np.where(
        df["status"].str.startswith(("1", "2")),
        np.maximum(df["eoq"], df["reorder_point"] + df["eoq"] - df["available_units"]),
        0,
    )
    return df


def kpis(df: pd.DataFrame) -> dict:
    w = df["annual_demand"]
    u, s = df["uniform_avg_stock_eur"].sum(), df["segment_avg_stock_eur"].sum()
    return {
        "uniform_avg_stock_eur": int(round(u)),
        "segmented_avg_stock_eur": int(round(s)),
        "saving_eur": int(round(u - s)),
        "saving_pct": float(round((u - s) / u * 100, 1)) if u else 0.0,
        "uniform_weighted_service_level_pct": float(round((df["uniform_sl"] * w).sum() / w.sum() * 100, 1)),
        "segmented_weighted_service_level_pct": float(round((df["segment_sl"] * w).sum() / w.sum() * 100, 1)),
        "parts_total": int(len(df)),
        "parts_by_status": {k: int(v) for k, v in df["status"].value_counts().sort_index().items()},
    }
