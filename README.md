# AI Inventory Copilot – spare parts planning with ABC/XYZ, safety stock and AI

A planning project for 300 spare parts (synthetic data, 52 weeks of demand) with three parts:
an Excel planning model, a Power BI dashboard and a Streamlit web app with an AI assistant.

**Live demo:** https://ai-inventory-copilot-prathamesh.streamlit.app/ (AI chat is password-protected; all other tabs are open)

**Main results**
- ABC/XYZ-segmented service levels reach **97% demand-weighted service** with about **121K € less average stock** than one 95% rule for every part.
- **953K €** of stock is above the planned maximum (excess stock).
- **29 parts** look fine by the reorder-point rule but run out before their open purchase order arrives.

## 1. Planning logic (same in Excel, Power BI and the app)
- **ABC** by annual consumption value (A ≤ 80%, B ≤ 95%), **XYZ** by demand variability (CV ≤ 0.5 / ≤ 1.0)
- **Service level per segment** (e.g. AX 95%, CZ 98%)
- **Safety stock** = z × std dev of weekly demand × √lead time
- **Reorder point** = avg weekly demand × lead time + safety stock
- **EOQ** = √(2 × annual demand × order cost / holding cost)
- **Status:** Critical (stock < safety stock), Reorder now (stock + open POs < reorder point),
  Watch (supplier confirmed PO late), Overstock

File: `Inventory_Policy_Optimizer.xlsx`

## 2. Planning web app with AI Copilot (Streamlit + Google Gemini API)

| Tab | Planner question it answers |
|---|---|
| Overview | Is our stock policy right? KPIs, inventory turns, days of supply, Pareto (ABC) chart |
| Reorder alerts | What do I need to order today? Including parts that look OK but run out before their open PO arrives |
| Stock-out risk | When will parts run out? Projection: stock + open POs − average demand |
| Part explorer | Why does this part have this safety stock? Projected stock chart with safety stock and reorder point |
| Excess & slow movers | How much capital is tied up above the planned maximum? |
| Supplier performance | Which suppliers deliver late? On-time delivery % vs 95% target |
| Forecast accuracy | How well can demand be forecast? Back-test of two methods (WMAPE, bias) by ABC/XYZ |
| AI Copilot | Questions in plain English or German, answered via function calling on the planning data |

The AI does not guess numbers: it calls planning functions (10 tools) and shows which data it used.
Supplier emails are drafts only; nothing is sent.
Example questions: see `DEMO_QUESTIONS.md`.

Files: `app.py`, `planning.py`, `analytics.py`, `copilot.py`, `data/`
(`po_history.csv` is a synthetic purchase order history for the supplier analysis.)

## 3. Power BI dashboard
File: `Inventory-Dashboard.pbix`. Three pages, each answering one planning question.

**Inventory policy – are we holding the right stock?**
Compares one service level for every part (95%) with segmented service levels.
A what-if slider shows the stock needed if every part got the same level (e.g. 99% ≈ +700K €).

<img width="1437" height="792" alt="page1_policy" src="https://github.com/user-attachments/assets/c5f31ca1-b145-41b9-8b90-6572eb462d13" />

**Reorder alerts – what needs action today?**
22 parts below safety stock, 74 to reorder now, 14 with late supplier confirmations;
suggested orders worth about 956K €. Clicking a supplier filters the page.

<img width="1441" height="787" alt="page2_alerts" src="https://github.com/user-attachments/assets/5e05596a-768c-4a51-8cf7-da0506adb619" />

**Part detail – why does this part have this safety stock?**
Drill-through page (right-click a part) showing segment, demand variability,
safety stock, reorder point, EOQ and the suggested order.

<img width="1432" height="792" alt="page3_details" src="https://github.com/user-attachments/assets/55dd75c9-7d76-44c5-bc05-e930281b878d" />

**Power BI features:** DAX measures, what-if parameter, drill-through,
conditional formatting, page navigation.

## 4. How this would run in a company
Stock, open purchase orders and supplier confirmations would come from SAP (e.g. MARD, EKKO/EKPO, EKES)
through a nightly extract. Power BI refreshes automatically every morning, so the alerts are current at
the start of the day. Safety stocks and reorder points are recalculated weekly or monthly, not daily,
to avoid nervous MRP; after the planner approves them they are updated in SAP (e.g. MM17).

## Tools
Excel · Power BI (DAX) · Python (pandas) · Streamlit · Google Gemini API

*All data is synthetic. No company data is used.*
