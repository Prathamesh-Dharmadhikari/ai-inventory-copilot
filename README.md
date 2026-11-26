# ai-inventory-copilot

## Power BI dashboard

Built on the Excel model above (file: `Inventory-Dashboard.pbix`, synthetic data).
Three pages, each answering one planning question:

**1. Inventory policy – are we holding the right stock?**
Compares one service level for every part (95%) with ABC/XYZ-segmented service levels.
Result: about 121K € less average stock at 97% demand-weighted service.
A what-if slider shows the stock needed if every part got the same service level (e.g. 99% ≈ +700K €).

<img width="1437" height="792" alt="page1_policy" src="https://github.com/user-attachments/assets/c5f31ca1-b145-41b9-8b90-6572eb462d13" />


**2. Reorder alerts – what needs action today?**
22 parts below safety stock, 74 to reorder now, 14 with late supplier confirmations;
suggested orders worth about 956K €. Clicking a supplier filters the page.

<img width="1441" height="787" alt="page2_alerts" src="https://github.com/user-attachments/assets/5e05596a-768c-4a51-8cf7-da0506adb619" />


**3. Part detail – why does this part have this safety stock?**
Drill-through page (right-click a part) showing segment, service level, demand variability,
safety stock, reorder point, EOQ and the suggested order.

<img width="1432" height="792" alt="page3_details" src="https://github.com/user-attachments/assets/55dd75c9-7d76-44c5-bc05-e930281b878d" />


**Power BI features used:** DAX measures, what-if parameter, drill-through,
conditional formatting, page navigation, synced slicers.
