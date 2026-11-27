"""AI planning copilot: Google Gemini with tools (function calling) that read the planning engine.

The model never invents numbers: every figure comes from a tool call
on the planning data. Emails are drafts only — a planner reviews and sends them.
"""
import json
import time

import requests
from statistics import NormalDist

import numpy as np
import pandas as pd

# Free-tier Gemini models. The first one that works is used; override with the GEMINI_MODEL secret.
MODEL_DEFAULT = "gemini-4-hydrogen"
MODEL_FALLBACKS = ["gemini-4-hydrogen", "gemini-3.8-flash", "gemini-flash-latest", "gemini-flash-lite-latest",
                   "gemini-2.5-flash", "gemini-2.5-flash-lite"]
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

SYSTEM_PROMPT = """You are an AI planning copilot for a material planner at a German manufacturing company.
You work on a spare-parts inventory model (300 parts, synthetic demo data) with ABC/XYZ segmentation,
safety stock, reorder points and EOQ lot sizes, plus a current stock snapshot with open purchase orders.

Rules:
- Get every number from the tools. Never guess or invent figures, part numbers or suppliers.
- Use as few tool calls as possible (the free AI quota is small): usually one or two calls are enough.
  If you need several tools, call them together in the same step. Do not look up parts one by one
  when list_parts already gives the needed columns.
- Be concise and practical, like an experienced planner talking to a colleague. Use € and units.
- When you explain a calculation, show the formula with the actual values
  (safety stock = z × σ weekly demand × √lead time; reorder point = avg weekly demand × lead time + safety stock;
  EOQ = √(2 × annual demand × ordering cost ÷ (unit cost × holding rate))).
- When asked for a supplier email: write it in formal German (Sie-form), clear subject line, list part number,
  description, quantity and required date or urgency, and ask for a confirmed delivery date. Mark it as a DRAFT
  for the planner to review. Never claim an email was sent.
- Mention limitations when relevant (normal-demand assumption is weak for intermittent Z-parts; data is synthetic).
- Answer in the language the user writes in, except supplier emails, which are German unless asked otherwise.
"""

TOOLS = [
    {
        "name": "get_portfolio_kpis",
        "description": "Overall KPIs: average stock value for uniform vs segmented policy, saving, weighted service levels, number of parts per stock status.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_part",
        "description": "All planning data and calculation details for one part number (e.g. MP-10001): demand statistics, ABC/XYZ segment, service level, safety stock, reorder point, EOQ, stock on hand, open PO, status.",
        "input_schema": {
            "type": "object",
            "properties": {"part": {"type": "string", "description": "Part number, e.g. MP-10001"}},
            "required": ["part"],
        },
    },
    {
        "name": "list_parts",
        "description": "List parts with optional filters, sorted. Use for questions like 'which A-parts are critical' or 'top overstock parts by value'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "abc": {"type": "string", "enum": ["A", "B", "C"]},
                "xyz": {"type": "string", "enum": ["X", "Y", "Z"]},
                "category": {"type": "string"},
                "supplier": {"type": "string", "description": "e.g. 'Supplier 03'"},
                "status_prefix": {"type": "string", "enum": ["1", "2", "3", "4", "5"],
                                  "description": "1 Critical, 2 Reorder now, 3 Watch (late PO), 4 OK, 5 Overstock"},
                "sort_by": {"type": "string",
                            "enum": ["annual_value_eur", "segment_avg_stock_eur", "weeks_of_cover", "cv", "suggested_order_units"]},
                "descending": {"type": "boolean"},
                "limit": {"type": "integer", "description": "max rows, default 10, max 50"},
            },
        },
    },
    {
        "name": "simulate_service_level",
        "description": "What-if: new safety stock, reorder point and average stock value for one part if its service level is changed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "part": {"type": "string"},
                "new_service_level": {"type": "number", "description": "between 0.5 and 0.999, e.g. 0.98"},
            },
            "required": ["part", "new_service_level"],
        },
    },
    {
        "name": "get_supplier_summary",
        "description": "Summary for one supplier: number of parts, parts by status, late purchase orders, parts that need an order now.",
        "input_schema": {
            "type": "object",
            "properties": {"supplier": {"type": "string", "description": "e.g. 'Supplier 03'"}},
            "required": ["supplier"],
        },
    },
]

PART_COLS = ["part", "description", "category", "supplier", "unit_cost_eur", "lead_time_weeks",
             "annual_demand", "avg_weekly", "std_weekly", "cv", "annual_value_eur", "abc", "xyz", "segment",
             "segment_sl", "segment_ss", "reorder_point", "eoq", "segment_avg_stock_eur",
             "on_hand_units", "open_po_units", "po_due_in_weeks", "po_confirmed_late", "weeks_of_cover",
             "status", "suggested_order_units"]


def _clean(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if isinstance(v, (np.floating, float)):
            out[k] = None if not np.isfinite(v) else round(float(v), 2)
        elif isinstance(v, np.integer):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def run_tool(name: str, args: dict, df: pd.DataFrame, params, kpi_fn) -> str:
    try:
        if name == "get_portfolio_kpis":
            return json.dumps(kpi_fn(df))

        if name == "get_part":
            row = df[df["part"].str.upper() == str(args["part"]).strip().upper()]
            if row.empty:
                return json.dumps({"error": f"Part {args['part']} not found. Part numbers run MP-10001 to MP-10300."})
            rec = _clean(row[PART_COLS].iloc[0].to_dict())
            rec["z_value"] = round(NormalDist().inv_cdf(rec["segment_sl"]), 3)
            rec["holding_rate"] = params.holding_rate
            rec["order_cost_eur"] = params.order_cost
            return json.dumps(rec)

        if name == "list_parts":
            q = df
            for col, key in (("abc", "abc"), ("xyz", "xyz")):
                if args.get(key):
                    q = q[q[col] == args[key]]
            if args.get("category"):
                q = q[q["category"].str.lower() == args["category"].lower()]
            if args.get("supplier"):
                q = q[q["supplier"].str.lower() == args["supplier"].lower()]
            if args.get("status_prefix"):
                q = q[q["status"].str.startswith(args["status_prefix"])]
            sort_by = args.get("sort_by", "annual_value_eur")
            q = q.sort_values(sort_by, ascending=not args.get("descending", True))
            limit = max(1, min(int(args.get("limit", 10)), 50))
            cols = ["part", "description", "supplier", "segment", "status", "on_hand_units", "reorder_point",
                    "open_po_units", "suggested_order_units", "weeks_of_cover", "segment_avg_stock_eur"]
            return json.dumps({"matches": int(len(q)),
                               "rows": [_clean(r) for r in q[cols].head(limit).to_dict("records")]})

        if name == "simulate_service_level":
            row = df[df["part"].str.upper() == str(args["part"]).strip().upper()]
            if row.empty:
                return json.dumps({"error": "Part not found"})
            r = row.iloc[0]
            sl = float(args["new_service_level"])
            if not 0.5 <= sl <= 0.999:
                return json.dumps({"error": "service level must be between 0.5 and 0.999"})
            new_ss = float(np.ceil(NormalDist().inv_cdf(sl) * r.std_weekly * np.sqrt(r.lead_time_weeks)))
            new_rop = float(np.ceil(r.avg_weekly * r.lead_time_weeks) + new_ss)
            new_val = (r.eoq / 2 + new_ss) * r.unit_cost_eur
            return json.dumps(_clean({
                "part": r.part, "current_service_level": r.segment_sl, "new_service_level": sl,
                "current_safety_stock": r.segment_ss, "new_safety_stock": new_ss,
                "current_reorder_point": r.reorder_point, "new_reorder_point": new_rop,
                "current_avg_stock_eur": r.segment_avg_stock_eur, "new_avg_stock_eur": new_val,
                "change_avg_stock_eur": new_val - r.segment_avg_stock_eur,
            }))

        if name == "get_supplier_summary":
            q = df[df["supplier"].str.lower() == str(args["supplier"]).lower()]
            if q.empty:
                return json.dumps({"error": "Supplier not found. Suppliers are 'Supplier 01' to 'Supplier 15'."})
            urgent = q[q["status"].str.startswith(("1", "2"))]
            late = q[q["po_confirmed_late"] == 1]
            cols = ["part", "description", "status", "on_hand_units", "reorder_point", "suggested_order_units",
                    "open_po_units", "po_due_in_weeks"]
            return json.dumps({
                "supplier": q["supplier"].iloc[0],
                "parts": int(len(q)),
                "parts_by_status": {k: int(v) for k, v in q["status"].value_counts().sort_index().items()},
                "late_purchase_orders": [_clean(r) for r in late[cols].to_dict("records")],
                "parts_to_order_now": [_clean(r) for r in urgent[cols].to_dict("records")],
            })

        return json.dumps({"error": f"unknown tool {name}"})
    except Exception as e:  # keep the chat alive and let the model explain
        return json.dumps({"error": str(e)})


def _gemini_schema(schema: dict) -> dict:
    """Convert a JSON schema to Gemini's schema format (upper-case types)."""
    out = {}
    for k, v in schema.items():
        if k == "type":
            out["type"] = v.upper()
        elif k == "properties":
            out["properties"] = {pk: _gemini_schema(pv) for pk, pv in v.items()}
        else:
            out[k] = v
    return out


def gemini_tools() -> list:
    decls = []
    for t in TOOLS:
        d = {"name": t["name"], "description": t["description"]}
        if t["input_schema"].get("properties"):  # Gemini rejects empty parameter objects
            d["parameters"] = _gemini_schema(t["input_schema"])
        decls.append(d)
    return [{"functionDeclarations": decls}]


class AIError(Exception):
    pass


class Busy(Exception):
    """Model temporarily overloaded (503/500) or rate-limited (429)."""


_no_thinking_cfg = set()  # models that reject the thinking setting


def _call(api_key: str, model: str, history: list) -> dict:
    gen = {"maxOutputTokens": 8192}
    if model not in _no_thinking_cfg:
        gen["thinkingConfig"] = {"thinkingLevel": "low"}  # faster answers, fewer timeouts
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": history,
        "tools": gemini_tools(),
        "generationConfig": gen,
    }
    try:
        r = requests.post(API_URL.format(model=model), json=body, timeout=60,
                          headers={"x-goog-api-key": api_key, "Content-Type": "application/json"})
    except requests.RequestException:
        raise Busy(f"{model}: timeout")
    if r.status_code == 400 and "thinking" in r.text.lower() and model not in _no_thinking_cfg:
        _no_thinking_cfg.add(model)  # older model: retry without the thinking setting
        return _call(api_key, model, history)
    if r.status_code == 429:
        raise Busy(f"{model}: quota")
    if r.status_code in (500, 503):
        time.sleep(3)
        try:
            r = requests.post(API_URL.format(model=model), json=body, timeout=60,
                              headers={"x-goog-api-key": api_key, "Content-Type": "application/json"})
        except requests.RequestException:
            raise Busy(f"{model}: timeout")
        if r.status_code in (429, 500, 503):
            raise Busy(f"{model}: {r.status_code}")
    if r.status_code == 404:
        raise FileNotFoundError(model)
    if r.status_code in (400, 401, 403) and "API key" in r.text:
        raise AIError("The Gemini API key is not valid. Check GEMINI_API_KEY in Streamlit → Settings → Secrets.")
    if r.status_code != 200:
        raise AIError(f"Gemini error {r.status_code}: {r.text[:300]}")
    return r.json()


def model_candidates(preferred: str | None) -> list:
    return ([preferred] if preferred else []) + [m for m in MODEL_FALLBACKS if m != preferred]


def pick_model(api_key: str, preferred: str | None) -> str:
    """Kept for compatibility: the first configured model; switching happens in chat()."""
    return model_candidates(preferred)[0]


def _call_any(api_key: str, models: list, history: list) -> dict:
    """Try the models in order; skip ones that don't exist or are overloaded."""
    busy = False
    for m in models:
        try:
            return _call(api_key, m, history)
        except FileNotFoundError:
            continue
        except Busy:
            busy = True
            continue
    if busy:
        raise AIError("The free AI quota is used up or Google's models are busy. Please try again in a minute; "
                      "if it keeps happening, the daily free limit is reached (it resets at about 9:00 German time).")
    raise AIError("No Gemini model found. Set GEMINI_MODEL in Secrets to a current model name from aistudio.google.com.")


def chat(api_key: str, model, history: list, df, params, kpi_fn, max_rounds: int = 6):
    """Run one user turn. `history` is the Gemini `contents` list; it is extended in place.
    Returns (final_text, tool_calls) where tool_calls is a list of (name, args) for transparency."""
    calls = []
    nudged = False
    for _ in range(max_rounds):
        models = model if isinstance(model, list) else model_candidates(model)
        data = _call_any(api_key, models, history)
        cands = data.get("candidates") or []
        if not cands or "content" not in cands[0]:
            reason = (cands[0].get("finishReason") if cands else None) or data.get("promptFeedback", {})
            return f"The AI gave no answer ({reason}). Please rephrase the question.", calls
        content = cands[0]["content"]
        content.setdefault("role", "model")
        history.append(content)  # keep the model turn unchanged (Gemini needs its thought signatures back)
        parts = content.get("parts", [])
        fcalls = [p["functionCall"] for p in parts if "functionCall" in p]
        if not fcalls:
            text = "\n".join(p["text"] for p in parts if "text" in p and not p.get("thought")).strip()
            if text:
                return text, calls
            if not nudged:  # model stopped without writing an answer: ask once for the final answer
                nudged = True
                if not parts:
                    history.pop()  # Gemini rejects empty turns in the history
                history.append({"role": "user", "parts": [{"text":
                    "Please write your final answer now for the planner, using the tool results above. "
                    "Keep it short: a table or bullet list of the parts and the quantities to order."}]})
                continue
            return ("The AI collected the data (see 'Data used') but did not write an answer. "
                    "Please ask again or make the question more specific."), calls
        responses = []
        for fc in fcalls:
            args = fc.get("args", {}) or {}
            calls.append((fc["name"], args))
            result = json.loads(run_tool(fc["name"], args, df, params, kpi_fn))
            fr = {"name": fc["name"], "response": {"result": result}}
            if fc.get("id"):
                fr["id"] = fc["id"]
            responses.append({"functionResponse": fr})
        history.append({"role": "user", "parts": responses})
    return "Stopped after too many tool calls — please ask a more specific question.", calls
