"""
Streamlit Investment Evaluator – Step 7 (LLM Picks + Allocation)

Includes Steps 1–6 and adds:
- LLM suggestions with allocations based on your investable amount (sidebar).
- Uses OPENAI_KEY = os.getenv("OPENAI_API_KEY").
- Headlines (2 per company) included in DataFrame for LLM reasoning.
"""
from dotenv import load_dotenv
import io
import math
import os
import re
from datetime import date, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import feedparser
from textblob import TextBlob
import streamlit as st

# ---------- Page config ----------
st.set_page_config(page_title="Nifty 500 Screener + LLM", page_icon="🤖", layout="wide")
load_dotenv()

# ---------- Cached fetch for Nifty 500 ----------
@st.cache_data(ttl=6*3600)
def fetch_nifty500_symbols():
    urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://www1.nseindia.com/content/indices/ind_nifty500list.csv",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=15)
            if r.ok and r.content:
                df = pd.read_csv(io.StringIO(r.content.decode("utf-8")))
                df.columns = df.columns.str.strip()
                df["YF_SYMBOL"] = df["Symbol"].astype(str).str.strip() + ".NS"
                return df[["Symbol", "Company Name", "Industry", "YF_SYMBOL"]]
        except Exception:
            pass
    return pd.DataFrame()

# ---------- Helper functions ----------
def normalize(value: float, min_v: float, max_v: float) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    if max_v == min_v:
        return 0.5
    clipped = max(min(value, max_v), min_v)
    return (clipped - min_v) / (max_v - min_v)


def composite_score(inputs: Dict[str, float], weights: Dict[str, float]) -> float:
    if not inputs or not weights:
        return 0.0
    wsum, total = 0.0, 0.0
    for k, v in inputs.items():
        w = weights.get(k, 0.0)
        wsum += w
        total += v * w
    return (total / wsum) if wsum else 0.0


@st.cache_data(ttl=6*3600)
def fetch_price_momentum(yf_symbol: str):
    try:
        end = date.today()
        start = end - timedelta(days=220)
        data = yf.download(yf_symbol, start=start, end=end, interval="1d", progress=False, threads=False)
        if data is None or data.empty:
            return None, None
        close = data.get('Adj Close') if 'Adj Close' in data else data.get('Close')
        if close is None or close.empty:
            return None, None
        close = close.dropna()
        mom_1m = float(close.pct_change(periods=21).iloc[-1] * 100) if len(close) > 21 else None
        mom_6m = float(close.pct_change(periods=126).iloc[-1] * 100) if len(close) > 126 else None
        return mom_1m, mom_6m
    except Exception:
        return None, None


@st.cache_data(ttl=6*3600)
def fetch_fundamentals(yf_symbol: str):
    try:
        t = yf.Ticker(yf_symbol)
        info = t.info
        revenue = info.get("totalRevenue")
        eps = info.get("trailingEps")
        pe = info.get("trailingPE")
        roe = info.get("returnOnEquity")
        debt = info.get("debtToEquity")
        return {
            "Revenue": revenue,
            "EPS": eps,
            "P/E": pe,
            "ROE": roe * 100 if roe else None,
            "Debt/Equity": debt,
        }
    except Exception:
        return {"Revenue": None, "EPS": None, "P/E": None, "ROE": None, "Debt/Equity": None}

@st.cache_data(ttl=1800)
def tool_prices(tickers, period="1mo"):
    """
    Fetch 1-month price & volume for all tickers.
    Returns dict: {ticker: DataFrame with Close, Volume}
    """
    out = {}
    for t in tickers:
        try:
            df = yf.download(t, period=period, interval="1d", progress=False, threads=False)
            if not df.empty:
                out[t] = df[["Close", "Volume"]].dropna()
        except Exception:
            pass
    return out

@st.cache_data(ttl=1800)
def fetch_price_data(symbols: List[str], period="1mo"):
    """
    Fetch 1-month close & volume data for multiple tickers efficiently.
    Returns dict: {symbol: (last_close, DataFrame)}.
    """
    out = {}
    try:
        data = yf.download(symbols, period=period, interval="1d", group_by="ticker", progress=False, threads=False)
    except Exception:
        return out

    # Case 1: Multi-ticker result (dict-like columns)
    for sym in symbols:
        try:
            if data is not None and sym in data and not data[sym].empty and "Close" in data[sym].columns:
                df = data[sym].dropna(subset=["Close"]).copy()
                last_close = float(df["Close"].iloc[-1])
                out[sym] = (last_close, df)
        except Exception:
            pass

    # Case 2: Fallback single-ticker mode
    if not out and isinstance(data, pd.DataFrame) and "Close" in data.columns:
        for sym in symbols:
            df = data.dropna(subset=["Close"]).copy()
            if not df.empty:
                last_close = float(df["Close"].iloc[-1])
                out[sym] = (last_close, df)
    return out


@st.cache_data(ttl=6*3600)
def fetch_news_headlines(company_name, limit=2):
    try:
        query = f"{company_name} stock site:moneycontrol.com OR site:reuters.com OR site:economictimes.indiatimes.com"
        feed_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(feed_url)
        headlines = []
        for entry in feed.entries[:limit]:
            title = re.sub(r"<.*?>", "", entry.title)
            link = entry.link
            headlines.append(f"{title} ({link})")
        return headlines
    except Exception:
        return []


def sentiment_from_headlines(headlines):
    if not headlines:
        return 0.0
    scores = [TextBlob(h).sentiment.polarity for h in headlines]
    return float(np.mean(scores)) if scores else 0.0

# ---------- NEW: LLM pick & allocation ----------

def _prepare_llm_payload(df: pd.DataFrame, top_k: int = 60) -> List[dict]:
    cols = [
        "Symbol", "Company", "Industry", "Score", "Sentiment", "1M%", "6M%",
        "P/E", "ROE%", "Debt/Equity", "EPS", "Headlines"
    ]
    present_cols = [c for c in cols if c in df.columns]
    sub = df[present_cols].head(top_k).fillna(0)
    return sub.to_dict(orient="records")


def llm_pick_and_allocate(df: pd.DataFrame, total_amount: float, strategy_note: str = "") -> tuple[str, dict]:
    """Ask LLM to select stocks and propose allocation. Returns (markdown, parsed_json_postprocessed)."""
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_KEY:
        return (
            "❌ OPENAI_API_KEY not set. Set it and rerun.\n"
            "export OPENAI_API_KEY=sk-...\n", {}
        )

    data_records = _prepare_llm_payload(df)

    system_prompt = (
        f"You are a meticulous long-term Indian equity analyst ({horizon}). "
        "Evaluate Nifty 500 stocks using fundamentals, momentum and headlines. "
        "Output a strictly valid JSON allocation plan (weights sum to ~100%), with INR amounts for the total investment."
    )
    user_prompt = {
        "objective": "Select 1–6 stocks and propose allocation % and INR amounts.",
        "total_amount_in_inr": total_amount,
        "strategy_note": strategy_note,
        "data_note": "Higher composite score is better; lower P/E & Debt/Equity better; higher ROE better; sentiment -1..+1.",
        "universe_top": data_records,
        "format": ("Return strict JSON with keys: "
                   "'picks' (list of {symbol, company, rationale, weight_pct, amount_in_inr}), 'why', 'risk_checks'.")
    }

    # ---- Call OpenAI (new style then legacy fallback)
    try:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": str(user_prompt)}
                ]
            )
            content = resp.choices[0].message.content
        except Exception:
            import openai
            openai.api_key = OPENAI_KEY
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                temperature=0.3,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": str(user_prompt)}
                ]
            )
            content = resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ LLM call failed: {e}", {}

    # ---- Clean & parse JSON (handles ```json fences)
    import json, re
    def _clean_json_text(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"^```[a-zA-Z]*", "", text.strip())
        text = re.sub(r"```$", "", text.strip())
        m = re.search(r"\{[\s\S]*\}", text)
        return m.group(0) if m else text.strip()

    try:
        clean_text = _clean_json_text(content)
        parsed = json.loads(clean_text)
    except Exception:
        return "❌ Could not parse LLM response.\nRaw:\n" + content, {}

    # ---- Postprocess picks: dedupe, clamp, renormalize to 100%, recompute INR
    picks = parsed.get("picks", [])
    by_symbol = {}
    for p in picks:
        sym = (p.get("symbol") or "").strip().upper()
        if not sym:
            continue
        if sym in by_symbol:
            # merge duplicates: keep first rationale, average weights (or keep max)
            by_symbol[sym]["weight_pct"] = max(by_symbol[sym]["weight_pct"], float(p.get("weight_pct", 0) or 0))
        else:
            by_symbol[sym] = {
                "symbol": sym,
                "company": p.get("company", ""),
                "rationale": p.get("rationale", ""),
                "weight_pct": float(p.get("weight_pct", 0) or 0),
            }

    # clamp negatives, drop zeros
    cleaned = [d for d in by_symbol.values() if d["weight_pct"] > 0]
    total_pct = sum(d["weight_pct"] for d in cleaned)
    if total_pct <= 0:
        # fallback: equal weights
        n = min(3, len(cleaned)) or min(3, len(by_symbol)) or 1
        cleaned = list(by_symbol.values())[:n]
        for d in cleaned: d["weight_pct"] = 100.0 / n
        total_pct = 100.0

    # renormalize to 100
    for d in cleaned:
        d["weight_pct"] = d["weight_pct"] * (100.0 / total_pct)
        d["amount_in_inr"] = round(total_amount * d["weight_pct"] / 100.0, 2)

    parsed_post = {
        "picks": cleaned,
        "why": parsed.get("why", ""),
        "risk_checks": parsed.get("risk_checks", "")
    }

    # ---- Markdown summary from postprocessed picks
    total_pct2 = sum(d["weight_pct"] for d in cleaned)
    md = ["### 🤖 LLM Picks & Allocation", ""]
    md = []
    md.append(f"Target investable amount: **₹{total_amount:,.0f}**")
    md.append(f"Number of picks: **{len(cleaned)}**, Normalized total weight: **{total_pct2:.2f}%**")
    md.append("")
    if cleaned:
        md.append("**Allocation:**")
        for d in cleaned:
            md.append(f"- **{d['symbol']} – {d['company']}**: **{d['weight_pct']:.2f}%** "
                      f"(₹{d['amount_in_inr']:,.0f}) — {d['rationale']}")
    if parsed_post.get("why"):
        why_text = parsed_post["why"]
        if isinstance(why_text, (dict, list)):
            why_text = json.dumps(why_text, indent=2)
        md.extend(["", "**Why these picks:**", str(why_text)])

    if parsed_post.get("risk_checks"):
        risk_text = parsed_post["risk_checks"]
        if isinstance(risk_text, (dict, list)):
            risk_text = json.dumps(risk_text, indent=2)
        md.extend(["", "**Risk checks:**", str(risk_text)])


    return "\n".join(md), parsed_post

def to_yf_symbol(nse_symbol: str) -> str:
    # Convert NSE tickers to Yahoo format
    s = (nse_symbol or "").upper()
    if s.endswith(".NS"):
        return s
    return s + ".NS"

@st.cache_data(ttl=3600)
def fetch_1m_close(yf_symbol: str, retries: int = 3, delay: float = 1.5):
    import time
    for _ in range(retries):
        try:
            data = yf.download(yf_symbol, period="6mo", interval="1d", progress=False, threads=False)
            if data is not None and not data.empty and "Close" in data:
                return data["Close"]
        except Exception:
            pass
        time.sleep(delay)
    return None


# ---------- Sidebar configuration ----------
st.sidebar.header("⚙️ Screener Settings")

horizon = st.sidebar.select_slider("Investment Horizon", options=["3Y", "5Y", "7Y", "10Y+"], value="10Y+")
review_date = st.sidebar.date_input("Review Date", value=date.today())
max_stocks_to_check = st.sidebar.slider("Max Stocks to Check", 10, 500, 10, step=5)

st.sidebar.markdown("---")
INV_AMT = st.sidebar.number_input("Investable Amount (₹)", value=100000.0, step=10000.0, min_value=0.0, help="Amount to allocate across LLM picks")
STRATEGY_NOTE = st.sidebar.text_area("Strategy note (optional)", value="Long-term, quality bias; avoid high leverage")

st.sidebar.markdown("---")
st.sidebar.subheader("Weights (drag to tune)")
weight_defaults = {
    "Valuation": 15,
    "Growth": 15,
    "Profitability": 15,
    "Financial Health": 15,
    "Moat": 10,
    "Management": 10,
    "Risk": 10,
    "Momentum": 5,
    "Flows": 5,
    "News": 5,
}
weights = {k: st.sidebar.slider(k, 0, 30, v) for k, v in weight_defaults.items()}


# ---------- Main title ----------
st.title("🤖 Nifty 500 Screener – Sentiment + LLM Allocation (Step 7)")
st.caption("Automatically evaluates Nifty 500, then asks an LLM to propose picks and allocation.")

# ---------- Fetch universe ----------
nifty_df = fetch_nifty500_symbols()
nifty_df = nifty_df.head(max_stocks_to_check)
if nifty_df.empty:
    st.error("⚠️ Could not fetch Nifty 500 list from NSE. Please retry later.")
    st.stop()

# ---------- Screener execution ----------
st.header("🔍 Run Screener")

if st.button("Run Screener for All Nifty 500 Stocks"):
    progress = st.progress(0)
    results = []
    n = len(nifty_df)

    for i, row in enumerate(nifty_df.itertuples(), start=1):
        sym = row.Symbol
        name = row._2
        ind = row.Industry
        yf_sym = row.YF_SYMBOL

        # Momentum
        mom1, mom6 = fetch_price_momentum(yf_sym)
        mom_score = 100 * normalize(((mom1 or 0) + (mom6 or 0)) / 2.0, -20, 20)

        # Fundamentals
        fundamentals = fetch_fundamentals(yf_sym)
        pe = fundamentals.get("P/E")
        roe = fundamentals.get("ROE")
        debt = fundamentals.get("Debt/Equity")
        eps = fundamentals.get("EPS")

        # Sentiment
        headlines = fetch_news_headlines(name, limit=2)
        sentiment_score = sentiment_from_headlines(headlines)
        news_score = 100 * normalize(sentiment_score, -1, 1)

        # Derived normalized fundamentals
        val_score = 100 * (1 - normalize(pe if pe else 0, 0, 50))
        prof_score = 100 * normalize(roe if roe else 0, 0, 30)
        fin_score = 100 * (1 - normalize(debt if debt else 0, 0, 300))
        gr_score = 100 * normalize(eps if eps else 0, 0, 100)

        moat_score, mgmt_score, risk_score, flows_score = [60]*4

        factors = {
            "Valuation": val_score/100,
            "Growth": gr_score/100,
            "Profitability": prof_score/100,
            "Financial Health": fin_score/100,
            "Moat": moat_score/100,
            "Management": mgmt_score/100,
            "Risk": risk_score/100,
            "Momentum": mom_score/100,
            "Flows": flows_score/100,
            "News": news_score/100,
        }

        score = composite_score(factors, weights)
        results.append({
            "Symbol": sym,
            "Company": name,
            "Industry": ind,
            "1M%": mom1,
            "6M%": mom6,
            "P/E": pe,
            "ROE%": roe,
            "Debt/Equity": debt,
            "EPS": eps,
            "Sentiment": round(sentiment_score, 2),
            "Headlines": " | ".join(headlines),
            "Score": round(score*100, 1)
        })
        progress.progress(i/n)

    df_all = pd.DataFrame(results).sort_values("Score", ascending=False)

    # ---------- 💰 Add live price details ----------
    st.info("Fetching latest prices and volume data for top stocks...")

    symbols = [to_yf_symbol(s) for s in df_all["Symbol"].tolist()]
    price_data = fetch_price_data(symbols, period="1mo")

    cur_prices, vol_avg, vol_change, mom_1m_calc = [], [], [], []

    for sym in df_all["Symbol"]:
        yf_sym = to_yf_symbol(sym)
        if yf_sym not in price_data:
            cur_prices.append(None)
            vol_avg.append(None)
            vol_change.append(None)
            # mom_1m_calc.append(None)
            continue

        try:
            last_close, df = price_data[yf_sym]
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df.dropna(subset=["Close"])

            # Compute average volume and change
            avg_vol = df["Volume"].mean() if "Volume" in df.columns else None
            if "Volume" in df.columns and len(df) > 10:
                early_vol = df["Volume"].iloc[:5].mean()
                late_vol = df["Volume"].iloc[-5:].mean()
                volchg = ((late_vol - early_vol) / early_vol * 100) if early_vol > 0 else None
            else:
                volchg = None

            # Compute 1M return
            mom1 = ((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100

            cur_prices.append(round(last_close, 2))
            vol_avg.append(round(avg_vol, 0) if avg_vol else None)
            vol_change.append(round(volchg, 1) if volchg else None)
            # mom_1m_calc.append(round(mom1, 2))
        except Exception:
            cur_prices.append(None)
            vol_avg.append(None)
            vol_change.append(None)
            # mom_1m_calc.append(None)

    df_all["Current Price"] = cur_prices
    df_all["Avg Volume"] = vol_avg
    df_all["Vol Change %"] = vol_change
    # df_all["1M Return % (Live)"] = mom_1m_calc

    st.dataframe(df_all.head(10), use_container_width=True)


    st.success(f"✅ Screener completed for {len(df_all)} stocks.")
    # st.subheader("📈 Top Stocks by Composite Score")
    # st.dataframe(df_all.head(20), use_container_width=True)

    csv = df_all.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download All Scores", csv, "nifty500_scores.csv", mime="text/csv")

    st.session_state["screener_df"] = df_all

# ---------- Results & LLM ----------
if "screener_df" in st.session_state:
    df_all = st.session_state["screener_df"]
    st.markdown("---")
    st.subheader("🤖 Ask LLM for Picks & Allocation")
    if st.button("Ask LLM for Suggestions"):
        with st.spinner("Contacting LLM and preparing allocation..."):
            md, parsed = llm_pick_and_allocate(df_all, INV_AMT, STRATEGY_NOTE)
            st.session_state["llm_md"] = md
            st.session_state["llm_parsed"] = parsed
            st.success("✅ LLM suggestions received and stored!")

    # ✅ Always display only from session state
    if "llm_md" in st.session_state:
        st.markdown(st.session_state["llm_md"])

    # Step 3: Separate button to show price charts for LLM picks
    if st.button("📊 Show Price Charts for Suggested Stocks"):
        parsed = st.session_state.get("llm_parsed", {})
        if parsed and parsed.get("picks"):
            st.markdown("### 📊 Price Charts for Suggested Stocks (Last 1 Month)")
            for d in parsed["picks"]:
                sym = d.get("symbol", "")
                comp = d.get("company", "")
                yf_sym = to_yf_symbol(sym)
                close = fetch_1m_close(yf_sym)

                # ✅ Skip invalid or empty data
                if close is None or len(close) < 2:
                    st.warning(f"Not enough data to plot for {sym}.")
                    continue

                # ✅ Handle both Series and DataFrame cases safely
                if isinstance(close, pd.DataFrame):
                    if "Close" in close.columns:
                        df_plot = close[["Close"]].copy()
                    else:
                        first_col = close.columns[0]
                        df_plot = close[[first_col]].rename(columns={first_col: "Close"})
                elif isinstance(close, pd.Series):
                    df_plot = close.to_frame(name="Close")
                else:
                    st.warning(f"Unexpected data format for {sym}.")
                    continue

                # ✅ Ensure datetime index
                df_plot.index = pd.to_datetime(df_plot.index, errors="coerce")
                df_plot = df_plot.dropna(subset=["Close"])
                if df_plot.empty:
                    st.warning(f"No valid data points for {sym}.")
                    continue

                # ✅ Render chart
                st.line_chart(df_plot, height=180)
                start_price = df_plot["Close"].iloc[0]
                end_price = df_plot["Close"].iloc[-1]
                change = ((end_price - start_price) / start_price) * 100
                trend = "🔼" if change >= 0 else "🔻"
                st.caption(f"**{sym} – {comp}** ({trend} {change:.1f}% over last 1 month)")
                st.markdown("---")
        else:
            st.info("Run 'Ask LLM for Suggestions' first.")

else:
    st.info("Click 'Run Screener for All Nifty 500 Stocks' to begin.")
