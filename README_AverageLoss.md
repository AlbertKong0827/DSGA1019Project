## Metric 2: Average Actual Loss by State (2018–2024)

This analysis calculates and visualizes the average actual loss per loan by U.S. state using Freddie Mac loan-level data from 2018 to 2024.

### 📌 Objective
To identify geographic patterns in loan losses and highlight high-risk regions by computing average actual loss from disposed/defaulted loans.

### 📁 Data Sources
- **Freddie Mac Single-Family Loan-Level Dataset**
- Servicing (`sample_svcg_YYYY.txt`) — provides actual loss data
- Origination (`sample_orig_YYYY.txt`) — provides loan details including `PROPERTY STATE`

### 🔍 Data Cleaning
- Extracted `ACTUAL LOSS CALCULATION` from `col_21` of the servicing file
- Merged with loan ID and `PROPERTY STATE` from origination (`col_19`, `col_16`)
- Filtered out `NaN` and zero-loss records
- Repeated for years 2018 to 2024

### 🧮 Computation
- Grouped all valid loans by `PROPERTY STATE`
- Calculated mean loss per state
- Sorted and visualized

### 📊 Visualization
A bar chart shows the average actual loss per loan by state, with state abbreviations and a two-column legend for clarity.

### 💡 Key Insights
- States like **Hawaii** and **Puerto Rico** show highest average losses
- Some states (e.g., **Wyoming**, **North Dakota**) exhibit minimal or even positive average values
- Losses vary sharply by geography, supporting the need for location-aware credit risk models

---
