# AI Inventory Copilot – spare parts planning with ABC/XYZ, safety stock and AI

A planning project for 300 spare parts (synthetic data, 52 weeks of demand) with three parts:
an Excel planning model, a Power BI dashboard and a web app with an AI assistant.

**Main result:** ABC/XYZ-segmented service levels reach 97% demand-weighted service
with about 121K € less average stock than one 95% rule for every part.

## 1. Planning logic (same in Excel, Power BI and the app)
- **ABC** by annual consumption value (A ≤ 80%, B ≤ 95%), **XYZ** by demand variability (CV ≤ 0.5 / ≤ 1.0)
- **Service level per segment** (e.g. AX 95%, CZ 98%)
- **Safety stock** = z × std dev of weekly demand × √lead time
- **Reorder point** = avg weekly demand × lead time + safety stock
- **EOQ** = √(2 × annual demand × order cost / holding cost)
- **Status:** Critical (stock < safety stock), Reorder now (stock + open POs < reorder point),
  Watch (supplier confirmed PO late), Overstock

File: `Inventory_Policy_Optimizer.xlsx`

## 2. AI Copilot web app (Streamlit + Claude API)
Planners can ask questions in plain language, e.g. *"Which parts are critical at Supplier 01?"*
The assistant answers by calling the planning functions (tool use), so the numbers come from
the calculation, not from guessing.

Files: `app.py`, `copilot.py`, `planning.py`, `data/`

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

## Tools
Excel · Power BI (DAX) · Python · Streamlit · Claude API

*All data is synthetic. No company data is used.*
