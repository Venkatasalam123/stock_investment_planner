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


def _format_snapshots_for_prompt(snapshots: Iterable[StockSnapshot]) -> str:
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
        
        # Forecast trend (CRITICAL)
        forecast_slope = getattr(snapshot, "forecast_slope", None)
        if forecast_slope is not None:
            trend = "BULLISH" if forecast_slope > 0 else "BEARISH"
            line_parts.append(f"6M Forecast={forecast_slope:.2%} ({trend})")
        
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
    investment_horizon: int,
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

    return f"""
You are an equity allocation assistant helping a retail investor in India.

Investor context:
- Investment horizon: {investment_horizon} years
- Investable amount: ₹{invest_amount:,.2f}
- Strategy notes: {strategy_notes or 'None'}

Market snapshots:
{_format_snapshots_for_prompt(snapshots)}

Recent news:
{_format_news_for_prompt(news_map)}

{_format_fii_trend(fii_trend)}

{_format_market_mood(market_mood)}

{evaluation_block}

CRITICAL RULES:
1. DO NOT recommend stocks with BEARISH 6M Forecast (negative forecast_slope) unless there are exceptional fundamental reasons AND the investor has a very long horizon (5+ years).
2. Prioritize stocks with positive forecast trends, strong technical indicators (RSI not overbought, bullish MACD, Golden Cross), and solid fundamentals.
3. If a stock shows BEARISH forecast but strong fundamentals, mention this contradiction clearly in risks and consider lower allocation or waiting.
4. MARKET MOOD STRATEGY (CRITICAL):
   - If market mood is "Fear" or "Extreme Fear" (index < 45): This is a BUYING OPPORTUNITY. Consider increasing allocation to quality stocks as fear often creates value. Be more aggressive in recommendations.
   - If market mood is "Greed" or "Extreme Greed" (index > 55): MARKET IS OVERVALUED. You MUST prioritize recommending to WAIT rather than buy. Only recommend stocks if there are exceptional value opportunities with very strong fundamentals (e.g., low P/E, high dividend yield, strong revenue growth) AND if the investor has a long horizon (5+ years). In most cases with "Greed" or "Extreme Greed", return an EMPTY allocations list and explain in guidance why waiting is the better option.
   - If market mood is "Neutral" (45-55): Proceed with standard analysis.
   - REMEMBER: If you say the market is in a state of greed/overvaluation, your guidance MUST recommend waiting or be very conservative. DO NOT contradict yourself by recommending multiple stocks while warning about overvaluation.
5. If the investor ALREADY OWNS the evaluation stock and its forecast is bearish, default to a **SELL / book profits** stance unless there is overwhelming long-term conviction.
6. Only recommend SELL when the realised gain is meaningful for a long-term investor (at least 5% in percentage terms and roughly ₹5,000 absolute). Otherwise recommend holding/accumulating or waiting.

DETAILED ANALYSIS REQUIRED:
- When recommending to WAIT, you MUST provide a detailed explanation that goes beyond just the market mood index.
- Include specific metrics that influenced your decision, such as:
  * Average/median P/E ratios across stocks (if >25, mention overvaluation)
  * RSI levels (if many stocks are overbought >70, mention technical overvaluation)
  * Distance from 52W highs (if stocks are near 52W highs, mention elevated valuations)
  * Forecast trends (if many stocks show bearish forecasts, mention this)
  * Beta/volatility levels (if high, mention market risk)
  * PEG ratios (if >1.5, mention growth not justifying valuation)
  * Debt levels (if high debt/equity ratios, mention financial stress)
  * Revenue/profit growth trends (if declining, mention fundamental weakness)
- Be specific: Instead of "market is overvalued", say "average P/E of 28 across analyzed stocks indicates overvaluation, with 60% of stocks trading within 5% of 52-week highs and RSI levels averaging 68, suggesting limited upside potential."
- Reference actual numbers from the data provided, not generic statements.

Analyse the data, decide whether to invest now, wait, or partially allocate. Return a JSON object with:
{{
  "summary": "<short summary>",
  "guidance": "<overall guidance with DETAILED metric analysis. If recommending wait, explain which specific metrics (P/E, RSI, 52W high distance, forecast trends, beta, etc.) influenced this decision. Include actual numbers from the data.>",
  "allocations": [
     {{
        "symbol": "<ticker>",
        "allocation_pct": <percentage of total, up to one decimal>,
        "rationale": "<why this stock, mention forecast trend and technical signals>",
        "risks": "<key risks, especially if forecast is bearish>"
     }}
  ]
}}

If recommending to wait entirely, set allocations to an empty list and explain why in guidance with SPECIFIC METRIC REFERENCES.
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
        temperature=0.2,
        max_tokens=900,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a financial planning assistant focusing on Indian equities. Provide structured JSON outputs and note uncertainties.",
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

    return content


def _parse_llm_json(raw: str, snapshots: Iterable[StockSnapshot]) -> LLMResult:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMServiceError("LLM did not return valid JSON") from exc

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
) -> tuple[LLMResult, str]:
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
    raw_response = _invoke_llm(prompt)
    parsed = _parse_llm_json(raw_response, snapshots)
    return parsed, raw_response

