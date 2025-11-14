# Data Flow Explanation

## When You Click "Run Investment Analysis"

### 1. **Data Extraction Phase** ✅
All 40 new metrics are extracted for each stock:

- **`fetch_snapshots()`** calls **`_fetch_snapshot()`** for each stock
- This extracts ALL metrics from:
  - yfinance ticker.info (valuation, ownership, timing)
  - yfinance financials (profitability, growth)
  - yfinance balance_sheet (debt, liquidity)
  - yfinance cashflow (cash flow metrics)
  - Price history calculations (technical, risk metrics)

**Result:** All 40 metrics are populated in `StockSnapshot` objects.

---

### 2. **LLM Analysis Phase** ✅
All extracted metrics are formatted and passed to the LLM:

- **`llm_pick_and_allocate()`** is called with all `snapshots`
- This calls **`_format_snapshots_for_prompt()`** which:
  - Formats ALL available metrics from each snapshot
  - Organizes them into sections:
    - **Tech:** RSI, MACD, MA, Volume, Beta, etc.
    - **Fund:** Valuation (P/E, PEG, P/B, P/S, EV/EBITDA), Profitability (ROE, ROIC, ROA, Margins), Growth, Efficiency, Debt, Cash Flow
    - **Adv Tech:** Stochastic, Williams %R, Support/Resistance, OBV
    - **Risk:** Sharpe, Sortino, Max Drawdown, Volatility
    - **Market:** Market Cap, Institutional Ownership, Float
    - **Timing:** Earnings Date, Ex-Dividend Date

**Result:** LLM receives comprehensive data for all 40 metrics (when available) for making investment recommendations.

---

### 3. **Database Storage Phase** ✅
All snapshot data is stored in the database:

- **`log_run()`** is called which:
  - Serializes all snapshots using `snapshot.to_dict()` 
  - This includes ALL 40 metrics in the serialized data
  - Stores in `snapshots_blob` column as JSON

**Result:** All metrics are permanently stored for future reference and performance tracking.

---

### 4. **UI Display Phase** ⚠️ **NeEDS UPDATE**
Currently, the UI table shows only OLD metrics:

**`show_snapshot_table()`** currently displays:
- Basic metrics: 1M/6M Change, Revenue, EPS, P/E, ROE, Debt/Equity
- Growth: Revenue Growth, Dividend Yield
- Cash Flow: Free Cash Flow, Operating Margin
- Technical: RSI, MA50/200, MACD, Bollinger Bands, Volume
- Forecast: 6M Forecast

**Missing from UI:** Most of the 40 new metrics are NOT displayed in the table yet.

---

### 5. **Performance Tracking Phase** 📊
Performance tracking uses minimal data:

- **`show_performance_tracking()`** uses `PredictionTracking` table
- This stores only:
  - Symbol
  - Suggested Price (from original analysis)
  - Current Price (fetched on demand)
  - Allocation %
  - Return % (calculated: (current - suggested) / suggested)
  - Days Since Suggestion

**Not used:** Most of the 40 new metrics are NOT used for performance tracking.
- Only prices are compared (suggested vs current)
- Risk metrics, debt metrics, etc. are not tracked over time

---

## Summary

| Phase | What Happens | All 40 Metrics? |
|-------|-------------|-----------------|
| **Data Extraction** | `_fetch_snapshot()` extracts all metrics | ✅ YES - All extracted |
| **LLM Analysis** | `_format_snapshots_for_prompt()` formats all for LLM | ✅ YES - All passed to LLM |
| **Database Storage** | `log_run()` serializes all snapshots | ✅ YES - All stored |
| **UI Display** | `show_snapshot_table()` shows metrics | ⚠️ NO - Only old metrics shown |
| **Performance Tracking** | `show_performance_tracking()` tracks prices | ⚠️ NO - Only prices tracked |

---

## Recommendations

1. **Update `show_snapshot_table()`** to include new metrics:
   - Add columns for: ROIC, ROA, PEG, P/B, P/S, Interest Coverage, Operating CF, Sharpe Ratio, etc.
   - Organize into expandable sections or tabs

2. **Enhance Performance Tracking** to include:
   - Track changes in fundamental metrics over time
   - Compare risk metrics at suggestion time vs now
   - Show how debt ratios, margins, etc. have changed

3. **Add Filters/Sorting** in UI table:
   - Sort by Sharpe Ratio, PEG, Interest Coverage, etc.
   - Filter stocks by specific metric thresholds

---

## Answer to Your Question

**Q: When I run "Run Investment Analysis", are these data going to be extracted and then passed to LLM?**

**A: YES! ✅**
- All 40 new metrics ARE extracted when you click "Run Investment Analysis"
- All 40 new metrics ARE passed to the LLM for recommendations
- All 40 new metrics ARE stored in the database

**Q: Or few are used for Performance analysis?**

**A: Partially ⚠️**
- Currently, Performance Tracking only uses prices (suggested vs current)
- Most of the 40 new metrics are NOT used for performance tracking yet
- They could be enhanced to track fundamental/risk metric changes over time

**However:**
- The UI table (`show_snapshot_table()`) currently does NOT display the new metrics
- You'll need to update the table to see all the new data

