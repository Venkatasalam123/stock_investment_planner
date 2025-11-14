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
        
        if tech_parts:
            line_parts.append(" | Tech: " + ", ".join(tech_parts))
        
        # Fundamentals
        fund_parts = []
        if fundamentals.get("totalRevenue"):
            fund_parts.append(f"Revenue={fundamentals.get('totalRevenue'):,}")
        if fundamentals.get("trailingEps") is not None:
            fund_parts.append(f"EPS={fundamentals.get('trailingEps'):.2f}")
        if fundamentals.get("trailingPE") is not None:
            fund_parts.append(f"P/E={fundamentals.get('trailingPE'):.2f}")
        if fundamentals.get("returnOnEquity") is not None:
            fund_parts.append(f"ROE={fundamentals.get('returnOnEquity'):.2%}")
        if snapshot.revenue_growth_yoy is not None:
            fund_parts.append(f"Rev Growth YoY={snapshot.revenue_growth_yoy:.2%}")
        if snapshot.dividend_yield is not None:
            fund_parts.append(f"Div Yield={snapshot.dividend_yield:.2%}")
        
        if fund_parts:
            line_parts.append(" | Fund: " + ", ".join(fund_parts))
        
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
4. MARKET MOOD STRATEGY:
   - If market mood is "Fear" or "Extreme Fear" (index < 45): This is a BUYING OPPORTUNITY. Consider increasing allocation to quality stocks as fear often creates value. Be more aggressive in recommendations.
   - If market mood is "Greed" or "Extreme Greed" (index > 55): Be CAUTIOUS. Market may be overvalued. Reduce allocation percentages, focus on defensive stocks, or recommend waiting for better entry points.
   - If market mood is "Neutral" (45-55): Proceed with standard analysis.
5. If the investor ALREADY OWNS the evaluation stock and its forecast is bearish, default to a **SELL / book profits** stance unless there is overwhelming long-term conviction.
6. Only recommend SELL when the realised gain is meaningful for a long-term investor (at least 5% in percentage terms and roughly ₹5,000 absolute). Otherwise recommend holding/accumulating or waiting.

Analyse the data, decide whether to invest now, wait, or partially allocate. Return a JSON object with:
{{
  "summary": "<short summary>",
  "guidance": "<overall guidance or wait conditions>",
  "allocations": [
     {{
        "symbol": "<ticker>",
        "allocation_pct": <percentage of total, up to one decimal>,
        "rationale": "<why this stock, mention forecast trend and technical signals>",
        "risks": "<key risks, especially if forecast is bearish>"
     }}
  ]
}}

If recommending to wait entirely, set allocations to an empty list and explain why in guidance.
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

