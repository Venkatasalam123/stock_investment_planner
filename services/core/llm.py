"""LLM integration for allocation suggestions using the OpenAI API."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from services.core.market_data import StockSnapshot
from services.core.market_mood import MarketMood
from services.core.news import NewsItem


class LLMServiceError(RuntimeError):
    """Raised when the LLM call fails or returns invalid output."""


@dataclass
class AllocationSuggestion:
    symbol: str
    allocation_pct: float
    rationale: str
    risks: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "allocation_pct": self.allocation_pct,
            "rationale": self.rationale,
            "risks": self.risks,
        }


@dataclass
class LLMResult:
    summary: str
    allocations: List[AllocationSuggestion]
    guidance: Optional[str] = None
    evaluation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "allocations": [allocation.to_dict() for allocation in self.allocations],
            "guidance": self.guidance,
            "evaluation": self.evaluation,
        }


def _format_snapshots_for_prompt(snapshots: Iterable[StockSnapshot], forecast_months: int = 6) -> str:
    lines: List[str] = []
    for snapshot in snapshots:
        fundamentals = snapshot.fundamentals or {}
        # Get current price from price_history if available
        history = getattr(snapshot, "price_history", None)
        close = None
        if history is not None and not history.empty and "Close" in history.columns:
            close = history["Close"].iloc[-1] if len(history) > 0 else None
        
        line_parts = [f"{snapshot.symbol} ({snapshot.short_name or 'N/A'}):"]
        
        # Price changes
        if snapshot.change_1m is not None:
            line_parts.append(f"1M Δ={snapshot.change_1m:.2%}")
        if snapshot.change_6m is not None:
            line_parts.append(f"6M Δ={snapshot.change_6m:.2%}")
        
        # Forecast trend (CRITICAL) – align wording with app thresholds
        forecast_slope = getattr(snapshot, "forecast_slope", None)
        if forecast_slope is not None:
            if forecast_slope <= -0.03:
                trend = "BEARISH"
            elif forecast_slope < 0.03:
                trend = "FLAT"
            elif forecast_slope < 0.08:
                trend = "BULLISH"
            else:
                trend = "STRONGLY BULLISH"
            # Format forecast period label dynamically
            if forecast_months < 12:
                period_label = f"{forecast_months}M"
            else:
                period_label = f"{forecast_months // 12}Y"
            line_parts.append(f"{period_label} Forecast={forecast_slope:.2%} ({trend})")
        
        # Technical indicators
        tech_parts = []
        if snapshot.rsi_14 is not None:
            rsi_status = "Overbought" if snapshot.rsi_14 > 70 else "Oversold" if snapshot.rsi_14 < 30 else "Neutral"
            tech_parts.append(f"RSI={snapshot.rsi_14:.1f} ({rsi_status})")
        if snapshot.moving_average_50 is not None and snapshot.moving_average_200 is not None:
            ma_signal = "Golden Cross" if snapshot.moving_average_50 > snapshot.moving_average_200 else "Death Cross"
            tech_parts.append(f"MA50/200: {ma_signal}")
        if snapshot.macd is not None and snapshot.macd_signal is not None:
            macd_signal = "Bullish" if snapshot.macd > snapshot.macd_signal else "Bearish"
            tech_parts.append(f"MACD: {macd_signal}")
        if snapshot.volume_ratio is not None:
            vol_status = "High" if snapshot.volume_ratio > 1.5 else "Low" if snapshot.volume_ratio < 0.5 else "Normal"
            tech_parts.append(f"Vol Ratio={snapshot.volume_ratio:.2f}x ({vol_status})")
        if snapshot.dist_52w_high is not None:
            tech_parts.append(f"52W High Dist={snapshot.dist_52w_high:.2%}")
        if snapshot.beta is not None:
            beta_status = "High Vol" if snapshot.beta > 1.2 else "Low Vol" if snapshot.beta < 0.8 else "Market Vol"
            tech_parts.append(f"Beta={snapshot.beta:.2f} ({beta_status})")
        
        if tech_parts:
            line_parts.append(" | Tech: " + ", ".join(tech_parts))
        
        # Fundamentals - Valuation
        fund_parts = []
        if fundamentals.get("totalRevenue"):
            fund_parts.append(f"Revenue={fundamentals.get('totalRevenue'):,}")
        if fundamentals.get("trailingEps") is not None:
            fund_parts.append(f"EPS={fundamentals.get('trailingEps'):.2f}")
        if fundamentals.get("trailingPE") is not None:
            fund_parts.append(f"P/E={fundamentals.get('trailingPE'):.2f}")
        if snapshot.peg_ratio is not None:
            fund_parts.append(f"PEG={snapshot.peg_ratio:.2f}")
        if snapshot.price_to_book is not None:
            fund_parts.append(f"P/B={snapshot.price_to_book:.2f}")
        if snapshot.price_to_sales is not None:
            fund_parts.append(f"P/S={snapshot.price_to_sales:.2f}")
        if snapshot.ev_to_ebitda is not None:
            fund_parts.append(f"EV/EBITDA={snapshot.ev_to_ebitda:.2f}")
        
        # Profitability
        if fundamentals.get("returnOnEquity") is not None:
            fund_parts.append(f"ROE={fundamentals.get('returnOnEquity'):.2%}")
        if snapshot.roic is not None:
            fund_parts.append(f"ROIC={snapshot.roic:.2%}")
        if snapshot.roa is not None:
            fund_parts.append(f"ROA={snapshot.roa:.2%}")
        if snapshot.gross_margin is not None:
            fund_parts.append(f"Gross Margin={snapshot.gross_margin:.2%}")
        if snapshot.net_margin is not None:
            fund_parts.append(f"Net Margin={snapshot.net_margin:.2%}")
        if snapshot.ebitda_margin is not None:
            fund_parts.append(f"EBITDA Margin={snapshot.ebitda_margin:.2%}")
        
        # Growth
        if snapshot.revenue_growth_yoy is not None:
            fund_parts.append(f"Rev Growth={snapshot.revenue_growth_yoy:.2%}")
        if snapshot.profit_growth_yoy is not None:
            fund_parts.append(f"Profit Growth={snapshot.profit_growth_yoy:.2%}")
        if snapshot.dividend_yield is not None:
            fund_parts.append(f"Div Yield={snapshot.dividend_yield:.2%}")
        
        # Efficiency & Liquidity
        if snapshot.asset_turnover is not None:
            fund_parts.append(f"Asset Turnover={snapshot.asset_turnover:.2f}")
        if snapshot.inventory_turnover is not None:
            fund_parts.append(f"Inv Turnover={snapshot.inventory_turnover:.2f}")
        if snapshot.current_ratio is not None:
            fund_parts.append(f"Current Ratio={snapshot.current_ratio:.2f}")
        if snapshot.quick_ratio is not None:
            fund_parts.append(f"Quick Ratio={snapshot.quick_ratio:.2f}")
        
        # Debt & Financial Health
        if snapshot.interest_coverage is not None:
            fund_parts.append(f"Interest Coverage={snapshot.interest_coverage:.2f}x")
        if snapshot.debt_to_assets is not None:
            fund_parts.append(f"Debt/Assets={snapshot.debt_to_assets:.2%}")
        if snapshot.working_capital is not None:
            if snapshot.working_capital >= 1e9:
                wc_str = f"₹{snapshot.working_capital/1e9:.2f}B"
            elif snapshot.working_capital >= 1e6:
                wc_str = f"₹{snapshot.working_capital/1e6:.2f}M"
            else:
                wc_str = f"₹{snapshot.working_capital:.0f}"
            fund_parts.append(f"Working Capital={wc_str}")
        
        # Cash Flow
        if snapshot.operating_cash_flow is not None:
            if abs(snapshot.operating_cash_flow) >= 1e9:
                ocf_str = f"₹{snapshot.operating_cash_flow/1e9:.2f}B"
            elif abs(snapshot.operating_cash_flow) >= 1e6:
                ocf_str = f"₹{snapshot.operating_cash_flow/1e6:.2f}M"
            else:
                ocf_str = f"₹{snapshot.operating_cash_flow:.0f}"
            fund_parts.append(f"Op CF={ocf_str}")
        if snapshot.capex is not None:
            if snapshot.capex >= 1e9:
                capex_str = f"₹{snapshot.capex/1e9:.2f}B"
            elif snapshot.capex >= 1e6:
                capex_str = f"₹{snapshot.capex/1e6:.2f}M"
            else:
                capex_str = f"₹{snapshot.capex:.0f}"
            fund_parts.append(f"CapEx={capex_str}")
        if snapshot.cash_flow_per_share is not None:
            fund_parts.append(f"CF/Share=₹{snapshot.cash_flow_per_share:.2f}")
        
        if fund_parts:
            line_parts.append(" | Fund: " + ", ".join(fund_parts))
        
        # Advanced Technical Indicators
        advanced_tech = []
        if snapshot.stochastic_k is not None and snapshot.stochastic_d is not None:
            stoch_status = "Overbought" if snapshot.stochastic_k > 80 else "Oversold" if snapshot.stochastic_k < 20 else "Neutral"
            advanced_tech.append(f"Stoch={snapshot.stochastic_k:.1f}/{snapshot.stochastic_d:.1f} ({stoch_status})")
        if snapshot.williams_r is not None:
            wr_status = "Oversold" if snapshot.williams_r < -80 else "Overbought" if snapshot.williams_r > -20 else "Neutral"
            advanced_tech.append(f"Williams%R={snapshot.williams_r:.1f} ({wr_status})")
        if snapshot.support_level is not None and snapshot.resistance_level is not None:
            if close is not None:
                advanced_tech.append(f"Support=₹{snapshot.support_level:.2f}, Resistance=₹{snapshot.resistance_level:.2f}")
        
        if advanced_tech:
            line_parts.append(" | Adv Tech: " + ", ".join(advanced_tech))
        
        # Risk Metrics
        risk_parts = []
        if snapshot.sharpe_ratio is not None:
            risk_parts.append(f"Sharpe={snapshot.sharpe_ratio:.2f}")
        if snapshot.sortino_ratio is not None:
            risk_parts.append(f"Sortino={snapshot.sortino_ratio:.2f}")
        if snapshot.max_drawdown is not None:
            risk_parts.append(f"Max DD={snapshot.max_drawdown:.2%}")
        if snapshot.volatility is not None:
            risk_parts.append(f"Volatility={snapshot.volatility:.2%}")
        
        if risk_parts:
            line_parts.append(" | Risk: " + ", ".join(risk_parts))
        
        # Market Cap & Ownership
        market_info = []
        if snapshot.market_cap is not None:
            if snapshot.market_cap >= 1e12:
                cap_str = f"₹{snapshot.market_cap/1e12:.2f}T"
            elif snapshot.market_cap >= 1e9:
                cap_str = f"₹{snapshot.market_cap/1e9:.2f}B"
            elif snapshot.market_cap >= 1e6:
                cap_str = f"₹{snapshot.market_cap/1e6:.2f}M"
            else:
                cap_str = f"₹{snapshot.market_cap:.0f}"
            market_info.append(f"Mkt Cap={cap_str}")
        if snapshot.institutional_ownership is not None:
            market_info.append(f"Inst Own={snapshot.institutional_ownership:.1%}")
        if snapshot.float_shares is not None:
            if snapshot.float_shares >= 1e9:
                float_str = f"{snapshot.float_shares/1e9:.2f}B"
            elif snapshot.float_shares >= 1e6:
                float_str = f"{snapshot.float_shares/1e6:.2f}M"
            else:
                float_str = f"{snapshot.float_shares:.0f}"
            market_info.append(f"Float={float_str} shares")
        
        if market_info:
            line_parts.append(" | Market: " + ", ".join(market_info))
        
        # Timing Factors
        timing_info = []
        if snapshot.earnings_date:
            timing_info.append(f"Earnings={snapshot.earnings_date}")
        if snapshot.ex_dividend_date:
            timing_info.append(f"Ex-Div={snapshot.ex_dividend_date}")
        
        if timing_info:
            line_parts.append(" | Timing: " + ", ".join(timing_info))
        
        lines.append(" - " + " ".join(line_parts))
    return "\n".join(lines[:100])  # cap to avoid overly long prompts


def _format_news_for_prompt(news_map: Dict[str, List[NewsItem]]) -> str:
    lines: List[str] = []
    for symbol, articles in news_map.items():
        if not articles:
            continue
        lines.append(f"{symbol} headlines:")
        for article in articles[:3]:
            lines.append(
                f"  - {article.title} ({article.source or 'Unknown'})"
            )
    return "\n".join(lines[:100])


def _format_fii_trend(fii_trend: Optional[pd.DataFrame]) -> str:
    if fii_trend is None or fii_trend.empty:
        return "No FII trend data available."
    latest = fii_trend.dropna(subset=["netBuySell"]).tail(6)
    lines = [f"{row['date'].date()}: {row['netBuySell']}" for _, row in latest.iterrows()]
    return "Recent FII net buy/sell (INR crores):\n" + "\n".join(lines)


def _format_market_mood(market_mood: Optional[MarketMood]) -> str:
    if market_mood is None:
        return "Market mood data not available."
    return f"""Market Mood Index: {market_mood.index}/100 ({market_mood.sentiment})
Description: {market_mood.description}
Recommendation: {market_mood.recommendation}"""


def _build_prompt(
    investment_horizon: int,  # Now in months
    invest_amount: float,
    strategy_notes: str,
    snapshots: Iterable[StockSnapshot],
    news_map: Dict[str, List[NewsItem]],
    fii_trend: Optional[pd.DataFrame],
    market_mood: Optional[MarketMood] = None,
    evaluation_symbol: Optional[str] = None,
    evaluation_position: Optional[str] = None,
    evaluation_shares: Optional[int] = None,
    evaluation_lots: Optional[List[Dict[str, Any]]] = None,
) -> str:
    evaluation_block = ""
    if evaluation_symbol:
        position_text = evaluation_position or "Do not own"
        shares_text = (
            f"Investor currently holds {evaluation_shares} share(s)."
            if evaluation_position == "Already own" and evaluation_shares
            else "Investor does not hold this stock yet."
        )
        lot_lines: List[str] = []
        if evaluation_lots:
            for lot in evaluation_lots:
                matched_date = lot.get("matched_date")
                matched_str = matched_date if matched_date else lot.get("requested_date")
                purchase_price = lot.get("purchase_price")
                if purchase_price:
                    lot_lines.append(
                        f"  - {matched_str}: {lot.get('shares', 0)} share(s) @ ₹{purchase_price:,.2f} (P/L: {lot.get('unrealized_pl', 0):+,.2f})"
                    )
                else:
                    lot_lines.append(
                        f"  - {lot.get('requested_date')}: {lot.get('shares', 0)} share(s) – price unavailable"
                    )
        lot_section = ("\nPurchase history:\n" + "\n".join(lot_lines)) if lot_lines else ""
        evaluation_block = f"""

Additional evaluation request:
- Target stock: {evaluation_symbol}
- Investor position: {position_text}
- {shares_text}{lot_section}

Provide a JSON object `stock_evaluation` with:
{{
  "symbol": "{evaluation_symbol}",
  "recommendation": "<buy now | wait | sell>",
  "reasoning": "<brief explanation referencing fundamentals, technicals, mood>",
  "confidence": "<percentage or qualitative>",
  "shares_to_sell": "<integer count of shares to sell now if selling is advised>"
}}
"""

    # Calculate forecast period: if horizon < 6 months, use horizon; otherwise use 6 months
    forecast_months = investment_horizon if investment_horizon < 6 else 6
    
    # Format forecast period label
    if forecast_months < 12:
        forecast_label = f"{forecast_months}M"
    else:
        forecast_label = f"{forecast_months // 12}Y"
    
    # Format horizon appropriately
    is_short_term = investment_horizon < 12
    if is_short_term:
        horizon_text = f"{investment_horizon} month{'s' if investment_horizon > 1 else ''}"
        investment_style = "SHORT-TERM"
        horizon_note = f"The investor aims to realize profits within {investment_horizon} month{'s' if investment_horizon > 1 else ''}. Focus on stocks with strong technical momentum, positive short-term catalysts, and the potential for quick price appreciation. Prioritize stocks with bullish near-term forecasts, favorable technical indicators (RSI, MACD, Golden Cross), and recent positive news that could drive price movement within this timeframe."
        rules_section = f"""1. For SHORT-TERM investments (horizon < 12 months), prioritize:
   - Strong technical momentum (bullish MACD, Golden Cross, RSI in favorable range 40-70)
   - Positive near-term catalysts from recent news
   - Bullish or STRONGLY BULLISH {forecast_label} Forecast (forecast_slope >= +0.03)
   - Stocks with high volume activity (volume_ratio > 1.2) suggesting active interest
   - AVOID stocks with BEARISH forecast, as they are unlikely to deliver profits within the short timeframe
2. For SHORT-TERM: Technical indicators and momentum are PRIMARY; fundamentals are SECONDARY. Focus on stocks that can deliver quick price appreciation.
3. Treat very small {forecast_label} forecast_slope values as NEUTRAL or only mildly positive, not bullish:
   - If -0.03 < forecast_slope < +0.03 (i.e. between -3% and +3% over {forecast_months} month{'s' if forecast_months > 1 else ''}), describe the trend as "flat/sideways" or "muted", not "bullish" or "strong positive".
   - When forecast_slope is in this flat range, explicitly mention the **limited upside** in the rationale (e.g., "{forecast_label} forecast is only +0.44%, effectively flat").
   - Only use the word "bullish" for forecast_slope >= +0.03, and "strongly bullish" for forecast_slope >= +0.08.
4. For SHORT-TERM: If a stock has BEARISH forecast, AVOID it completely unless there is strong technical breakout potential and positive news catalyst."""
    else:
        years = investment_horizon / 12
        if years == int(years):
            horizon_text = f"{int(years)} year{'s' if years > 1 else ''}"
        else:
            horizon_text = f"{years:.1f} year{'s' if years > 1 else ''}"
        investment_style = "LONG-TERM"
        horizon_note = f"The investor has a {horizon_text} investment horizon. Focus on fundamental value, sustainable growth, and long-term compounding potential. Consider P/E ratios, ROE, ROIC, debt levels, and consistent revenue/profit growth. Technical indicators are secondary to fundamental strength for long-term wealth building."
        rules_section = f"""1. For LONG-TERM investments, prioritize:
   - Strong fundamentals (ROE > 15%, ROIC > 10%, positive growth)
   - Consistent revenue and profit growth
   - Reasonable valuation (P/E < 25 for growth stocks, PEG < 1.5)
   - Low debt levels and positive cash flow
   - DO NOT recommend stocks with BEARISH {forecast_label} Forecast unless there are exceptional fundamental reasons AND the investor has a very long horizon (5+ years)
2. For LONG-TERM: Fundamentals and valuation are PRIMARY; technical indicators are SECONDARY. Focus on stocks with sustainable competitive advantages.
3. Treat very small {forecast_label} forecast_slope values as NEUTRAL or only mildly positive, not bullish:
   - If -0.03 < forecast_slope < +0.03 (i.e. between -3% and +3% over {forecast_months} month{'s' if forecast_months > 1 else ''}), describe the trend as "flat/sideways" or "muted", not "bullish" or "strong positive".
   - When forecast_slope is in this flat range, explicitly mention the **limited upside** in the rationale (e.g., "{forecast_label} forecast is only +0.44%, effectively flat").
   - Only use the word "bullish" for forecast_slope >= +0.03, and "strongly bullish" for forecast_slope >= +0.08.
4. For LONG-TERM: If a stock shows BEARISH forecast but strong fundamentals, mention this contradiction clearly in risks and consider lower allocation or waiting."""
    
    return f"""
You are an equity allocation assistant helping a retail investor in India.

Investor context:
- Investment horizon: {horizon_text} ({investment_style} investment)
- Investable amount: ₹{invest_amount:,.2f}
- Strategy notes: {strategy_notes or 'None'}

{horizon_note}

Market snapshots:
{_format_snapshots_for_prompt(snapshots, forecast_months=forecast_months)}

Recent news:
{_format_news_for_prompt(news_map)}

{_format_fii_trend(fii_trend)}

{_format_market_mood(market_mood)}

{evaluation_block}

CRITICAL RULES:
{rules_section}

STOCK-BY-STOCK EVALUATION APPROACH:
- EVALUATE EACH STOCK INDIVIDUALLY based on its own merits, not market-wide averages.
- Market mood is ONE FACTOR among many (fundamentals, technicals, valuation, growth) - NOT the sole deciding factor.
- Consider market mood as CONTEXT, not a hard rule:
  * Fear/Extreme Fear (< 45): Favorable for finding undervalued opportunities, but still evaluate each stock's fundamentals.
  * Greed/Extreme Greed (> 55): Exercise more caution and look for stocks with strong value propositions (low P/E, good PEG, strong fundamentals), but DO NOT automatically wait. Many quality stocks can still be good investments even in a greedy market.
  * Neutral (45-55): Standard evaluation criteria apply.
- For each stock, check:
  * Is the forecast trend positive (bullish)?
  * Are technical indicators favorable (RSI not overbought, bullish signals)?
  * Is valuation reasonable (P/E < 25 for growth stocks, PEG < 1.5, P/B reasonable)?
  * Are fundamentals strong (ROE > 15%, ROIC > 10%, positive revenue/profit growth)?
  * Is the stock trading at a reasonable distance from 52W high (< 20% below suggests not extremely overvalued)?
  * Are there any red flags (high debt, negative cash flow, declining growth)?
- Market mood should influence ALLOCATION SIZE (smaller during greed, larger during fear) but should not prevent you from recommending genuinely good individual stocks.

4. If the investor ALREADY OWNS the evaluation stock and its forecast is bearish, default to a **SELL / book profits** stance unless there is overwhelming long-term conviction.
5. Only recommend SELL when the realised gain is meaningful for a long-term investor (at least 5% in percentage terms and roughly ₹5,000 absolute). Otherwise recommend holding/accumulating or waiting.

DETAILED ANALYSIS REQUIRED:
- When recommending to WAIT for specific stocks, provide individual stock-level analysis.
- Reference SPECIFIC METRICS FOR EACH STOCK, not averages across all stocks.
- Example: "Stock ABC has P/E of 32 (overvalued), RSI of 75 (overbought), and is trading at 2% below 52W high. Stock XYZ has P/E of 18 (fair), RSI of 45 (neutral), and strong fundamentals - recommended despite overall market greed."
- When market mood is Greed/Extreme Greed, mention it as context but focus on identifying stocks with individual value propositions that overcome the general market condition.
- Be specific with actual numbers from the data provided for EACH stock you evaluate, not generic market-wide statements.

Analyse the data, decide whether to invest now, wait, or partially allocate. Return a JSON object with:
{{
  "summary": "<short summary>",
  "guidance": "<overall guidance with INDIVIDUAL STOCK-LEVEL analysis. Evaluate each stock on its own merits. If recommending wait for certain stocks, explain which specific metrics FOR EACH STOCK (P/E, RSI, 52W high distance, forecast trends, beta, etc.) influenced this decision. Include actual numbers from the data for individual stocks, not market-wide averages. Market mood should be mentioned as context but not as the sole reason to wait.>",
  "allocations": [
     {{
        "symbol": "<ticker>",
        "allocation_pct": <percentage of total, up to one decimal>,
        "rationale": "<why this stock, mention forecast trend and technical signals>",
        "risks": "<key risks, especially if forecast is bearish>"
     }}
  ]
}}

If recommending to wait entirely (rare - only if truly no stocks meet quality criteria), set allocations to an empty list and explain why in guidance with SPECIFIC STOCK-LEVEL METRIC REFERENCES for each evaluated stock.
Ensure allocation percentages sum to <= 100. Use concise language.
If no specific stock evaluation is requested, set "stock_evaluation": null.
"""


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMServiceError("OPENAI_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type(Exception),
)
def _invoke_llm(prompt: str) -> str:
    client = _client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,  # Make outputs as deterministic as possible
        seed=42,  # Fix a seed so identical prompts return identical outputs (best-effort)
        max_tokens=2000,  # Allow detailed stock-by-stock analysis
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a financial planning assistant focusing on Indian equities. You MUST return valid JSON only, no markdown, no code blocks, just raw JSON. Provide structured JSON outputs and note uncertainties.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError) as exc:
        raise LLMServiceError("Unexpected response format from OpenAI Chat Completions API") from exc

    if not content:
        raise LLMServiceError("LLM returned empty response.")

    # Extract token usage information (best-effort; fields may differ by SDK version)
    usage_dict: Dict[str, Any] | None = None
    usage = getattr(response, "usage", None)
    if usage is not None:
        try:
            # Newer SDKs: input_tokens / output_tokens / total_tokens
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)

            # Fallback for older naming (prompt_tokens / completion_tokens)
            if input_tokens is None:
                input_tokens = getattr(usage, "prompt_tokens", None)
            if output_tokens is None:
                output_tokens = getattr(usage, "completion_tokens", None)

            usage_dict = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
        except Exception:
            usage_dict = None

    # Return both the content and usage in a serializable wrapper
    return json.dumps({"content": content, "usage": usage_dict}, default=str)


def _parse_llm_json(raw: str, snapshots: Iterable[StockSnapshot]) -> LLMResult:
    """Parse LLM JSON response with fallback handling for markdown-wrapped JSON."""
    # Try to extract JSON from markdown code blocks if present
    cleaned_raw = raw.strip()
    
    # Remove markdown code block markers if present
    if cleaned_raw.startswith("```json"):
        cleaned_raw = cleaned_raw[7:]  # Remove ```json
    elif cleaned_raw.startswith("```"):
        cleaned_raw = cleaned_raw[3:]  # Remove ```
    
    if cleaned_raw.endswith("```"):
        cleaned_raw = cleaned_raw[:-3]  # Remove trailing ```
    
    cleaned_raw = cleaned_raw.strip()
    
    # Try to find JSON object boundaries if wrapped in text
    if not cleaned_raw.startswith("{"):
        # Try to find the first {
        start_idx = cleaned_raw.find("{")
        if start_idx >= 0:
            cleaned_raw = cleaned_raw[start_idx:]
    
    # Try to find the last } if JSON is embedded
    if cleaned_raw.count("{") > cleaned_raw.count("}"):
        # Unbalanced, try to fix by finding last }
        last_brace = cleaned_raw.rfind("}")
        if last_brace >= 0:
            cleaned_raw = cleaned_raw[:last_brace + 1]
    
    try:
        parsed = json.loads(cleaned_raw)
    except json.JSONDecodeError as exc:
        # Include first 500 chars of raw response in error for debugging
        error_msg = f"LLM did not return valid JSON. Raw response (first 500 chars): {raw[:500]}"
        # Also include the cleaned version for comparison
        if cleaned_raw != raw[:500]:
            error_msg += f" | Cleaned version (first 500 chars): {cleaned_raw[:500]}"
        raise LLMServiceError(error_msg) from exc

    summary = parsed.get("summary") or "No summary provided."
    guidance = parsed.get("guidance")
    allocations_payload = parsed.get("allocations", [])

    # Build snapshot map for filtering
    snapshot_map = {snapshot.symbol: snapshot for snapshot in snapshots}
    
    allocations: List[AllocationSuggestion] = []
    filtered_out: List[str] = []
    
    for item in allocations_payload:
        try:
            symbol = item.get("symbol", "")
            snapshot = snapshot_map.get(symbol)
            
            # Filter out stocks with strongly negative forecasts (unless explicitly justified)
            forecast_slope = getattr(snapshot, "forecast_slope", None) if snapshot else None
            if snapshot and forecast_slope is not None and forecast_slope < -0.10:
                filtered_out.append(f"{symbol} (forecast: {forecast_slope:.2%})")
                continue
            
            allocations.append(
                AllocationSuggestion(
                    symbol=symbol,
                    allocation_pct=float(item.get("allocation_pct", 0.0)),
                    rationale=item.get("rationale", ""),
                    risks=item.get("risks", ""),
                )
            )
        except (TypeError, ValueError):
            continue

    # Update guidance if we filtered out stocks
    if filtered_out:
        filter_note = f" Note: Filtered out {len(filtered_out)} stock(s) with strongly negative forecasts: {', '.join(filtered_out)}."
        if guidance:
            guidance += filter_note
        else:
            guidance = filter_note.strip()

    evaluation = parsed.get("stock_evaluation")
    if evaluation is not None and not isinstance(evaluation, dict):
        evaluation = None
    if evaluation is not None and evaluation.get("shares_to_sell") is not None:
        try:
            evaluation["shares_to_sell"] = int(float(evaluation["shares_to_sell"]))
        except (ValueError, TypeError):
            pass

    return LLMResult(summary=summary, guidance=guidance, allocations=allocations, evaluation=evaluation)


def llm_pick_and_allocate(
    investment_horizon: int,
    invest_amount: float,
    strategy_notes: str,
    snapshots: Iterable[StockSnapshot],
    news_map: Dict[str, List[NewsItem]],
    fii_trend: Optional[pd.DataFrame] = None,
    market_mood: Optional[MarketMood] = None,
    evaluation_symbol: Optional[str] = None,
    evaluation_position: Optional[str] = None,
    evaluation_shares: Optional[int] = None,
    evaluation_lots: Optional[List[Dict[str, Any]]] = None,
) -> tuple[LLMResult, str, Optional[Dict[str, Any]]]:
    """Call the LLM to generate allocation suggestions.

    Returns the parsed `LLMResult` alongside the raw JSON string emitted by the model.
    """
    prompt = _build_prompt(
        investment_horizon=investment_horizon,
        invest_amount=invest_amount,
        strategy_notes=strategy_notes,
        snapshots=snapshots,
        news_map=news_map,
        fii_trend=fii_trend,
        market_mood=market_mood,
        evaluation_symbol=evaluation_symbol,
        evaluation_position=evaluation_position,
        evaluation_shares=evaluation_shares,
        evaluation_lots=evaluation_lots,
    )
    wrapped = _invoke_llm(prompt)

    # Unwrap content and usage from the JSON wrapper
    try:
        payload = json.loads(wrapped)
        raw_response = payload.get("content", "")
        usage = payload.get("usage")
    except Exception:
        # Fallback: treat entire string as raw response if wrapper parsing fails
        raw_response = wrapped
        usage = None

    parsed = _parse_llm_json(raw_response, snapshots)
    return parsed, raw_response, usage

