# AI Planning Copilot – demo questions

The AI Copilot answers questions in plain English or German. It does not guess numbers:
for every question it looks up the planning data with tool calls (open **"Data used"** under
an answer to see which data it used).

Data: 300 spare parts (MP-10001 to MP-10300), 15 suppliers (Supplier 01 to Supplier 15),
8 categories (Electronic modules, Sensors, Hydraulic valves, Seals, Filters, Belts, Fasteners, Bearings).
All data is synthetic.

---

## 7-question interview demo (in this order)

1. **"Give me a short overview of our inventory situation."**
   → KPIs: stock value, savings from segmentation, service level, parts per status
2. **"Which parts are critical right now, and what should I order?"**
   → prioritised order list, the same numbers as the Power BI dashboard
3. **"Why is MP-10024 critical? Show me the calculation."**
   → safety stock, reorder point and EOQ formulas with the actual values
4. **"What happens to MP-10024 if I raise its service level to 99%?"**
   → what-if: new safety stock, reorder point and extra stock value in €
5. **"Which parts look OK but run out before their open PO arrives?"**
   → hidden risks that a simple status rule misses
6. **"Which suppliers have the worst on-time delivery?"**
   → OTD % ranking against the 95% target
7. **"Schreibe eine Mahnung an Supplier 01 für die verspäteten und dringenden Teile."**
   → German supplier email (draft only, nothing is sent)

Then open **"Data used"** to show that every number came from the planning model.

---

## 1. Overview and KPIs
- Give me a short overview of our inventory situation.
- How much average stock do we save with ABC/XYZ segmentation compared with 95% for every part?
- What is our demand-weighted service level with the segmented policy?
- How many parts are critical, to reorder, on watch, OK and overstocked?
- Is the segmented policy better than one service level for all parts? Explain why.

## 2. Critical parts and what to order
- Which parts are critical right now, and what should I order?
- Which A-parts need to be ordered now?
- Show the 10 critical parts with the lowest weeks of cover.
- Which parts need an order now and have no open purchase order?
- What is the total suggested order quantity for the critical parts?
- Which critical parts already have an open PO that I should expedite?

## 3. Filter and rank parts
- Which sensors need to be ordered now?
- Show all critical parts in the category Bearings.
- Which AX parts are below their reorder point?
- Which parts have the highest demand variability (CV)?
- Show the top 10 overstock parts by stock value.
- Which parts have the most weeks of cover?
- Which C-parts tie up the most stock value?
- List the Z-parts (irregular demand) that are critical.

## 4. One part in detail
- Explain why MP-10001 has this safety stock.
- Why is MP-10024 critical? Show me the calculation.
- What is the reorder point of MP-10188, and how is it calculated?
- Which ABC/XYZ segment is MP-10061 in, and what does that mean?
- How was the order quantity (EOQ) for MP-10013 calculated?
- How many weeks of cover does MP-10126 have?

## 5. What-if for one part
- What happens to MP-10024 if I raise its service level to 99%?
- How much stock do I save if MP-10013 goes down to 90% service level?
- Compare MP-10061 at 95% and at 98% service level.
- Is it worth raising MP-10188 to 99%? How much extra stock in € would that cost?

## 6. Suppliers
- Which suppliers cause the most critical parts?
- Give me a summary of Supplier 05.
- Which purchase orders from Supplier 11 are confirmed late?
- Which parts do I need to order from Supplier 01 now?
- Which supplier should I call first today, and why?

## 7. Supplier emails (drafts only, nothing is sent)
- Draft a German reminder email to Supplier 03 for their late and urgent parts.
- Schreibe eine Bestellung an Supplier 11 für die dringenden Teile.
- Draft an email to Supplier 05 asking them to deliver the open PO for MP-10013 earlier.
- Write the reminder to Supplier 01 in English.

## 8. Questions in German
- Welche Teile sind aktuell kritisch und was soll ich bestellen?
- Erkläre mir den Sicherheitsbestand von MP-10024.
- Welche Lieferanten verursachen die meisten kritischen Teile?
- Was passiert, wenn ich bei MP-10061 den Servicegrad auf 99 % erhöhe?

## 9. When will we run out? (stock projection)
- When will MP-10024 run out if I don't order?
- Show the projected stock of MP-10113 for the next 8 weeks.
- Which A-parts run out within the next 4 weeks?
- Which parts look OK but run out before their open PO arrives?
- If I place the suggested order for MP-10044 today, will it arrive in time?

## 10. Excess and slow-moving stock
- How much capital is tied up in excess stock?
- Show the top 10 parts by excess value.
- Which sensors have excess stock?
- Which parts are slow-moving or have falling demand?
- What are our inventory turns and days of supply, and what is the target?

## 11. Supplier performance
- Which suppliers have the worst on-time delivery?
- What is the on-time delivery rate of Supplier 01, and how late are they on average?
- Which suppliers are below the 95% target and also cause critical parts?

## 12. Forecast accuracy
- How accurate would a simple forecast be for our parts?
- Is the forecast better for X-parts or Z-parts? Why?
- Which A-parts have the highest forecast error?
- Which forecast method works better for MP-10002?

## 13. Method and limitations
- Explain ABC/XYZ classification in simple terms.
- Why do Z-parts get a high service level in this model?
- What are the limitations of the safety stock formula for irregular demand?
- Why do some cheap parts get more stock with the segmented policy?

---

## What the Copilot does not do
- **Produce a real demand forecast**: it only back-tests two simple methods to measure forecast accuracy.
- **Show week-by-week demand history**: use the chart in the *Part explorer* tab.
- **What-if for all parts at once**: use the service level sliders in the sidebar (or the Power BI slider).
- **Answer from outside the data** (real prices, real suppliers): it says it doesn't have that instead of guessing.
- **Place orders or send emails**: emails are drafts for the planner to review.
