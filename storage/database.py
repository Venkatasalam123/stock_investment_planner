"""SQLite persistence helpers for tracking Streamlit app executions."""

from __future__ import annotations

import json
import os
import pathlib
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import DateTime, Float, ForeignKey, String, Text, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy import inspect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker, selectinload

from services.core.market_data import StockSnapshot
from services.core.market_mood import MarketMood
from services.core.news import NewsItem
from services.core.llm import LLMResult


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/app.db")


def _ensure_sqlite_directory(url: str) -> None:
    if url.startswith("sqlite:///"):
        db_path = url.replace("sqlite:///", "")
        path = pathlib.Path(db_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)


class Base(DeclarativeBase):
    pass


class RunRecord(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    horizon_years: Mapped[int] = mapped_column()
    stock_universe: Mapped[int] = mapped_column()
    invest_amount: Mapped[float] = mapped_column(Float)
    strategy_notes: Mapped[str] = mapped_column(Text, default="")
    llm_summary: Mapped[str] = mapped_column(Text)
    llm_guidance: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    llm_raw: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    snapshots_blob: Mapped[str] = mapped_column(Text)
    news_blob: Mapped[str] = mapped_column(Text)
    fii_blob: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    universe_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    custom_symbols_blob: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    market_mood_index: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    market_mood_sentiment: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    market_mood_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    market_mood_recommendation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    stock_evaluation_blob: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    suggestions: Mapped[List["SuggestionRecord"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )
    predictions: Mapped[List["PredictionTracking"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )


class SuggestionRecord(Base):
    __tablename__ = "suggestions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    symbol: Mapped[str] = mapped_column(String(32))
    allocation_pct: Mapped[float] = mapped_column(Float)
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    risks: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    run: Mapped[RunRecord] = relationship(back_populates="suggestions")


class PredictionTracking(Base):
    __tablename__ = "prediction_tracking"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    suggested_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    suggested_price: Mapped[float] = mapped_column(Float, nullable=False)
    allocation_pct: Mapped[float] = mapped_column(Float, nullable=False)
    action_taken: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)  # "bought", "wait", "skipped"
    current_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_price_updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    return_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    days_since_suggestion: Mapped[Optional[int]] = mapped_column(nullable=True)
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    risks: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Fundamental metrics at suggestion time
    suggested_pe: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    suggested_peg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    suggested_roe: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    suggested_roic: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    suggested_debt_to_equity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    suggested_interest_coverage: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    suggested_revenue_growth: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    suggested_profit_growth: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    suggested_beta: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Current fundamental metrics (updated when prices are updated)
    current_pe: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_peg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_roe: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_roic: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_debt_to_equity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_interest_coverage: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_revenue_growth: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_profit_growth: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_beta: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_metrics_updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    run: Mapped[RunRecord] = relationship(back_populates="predictions")


_ensure_sqlite_directory(DATABASE_URL)
engine: Engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)
    _ensure_additional_columns()


def _ensure_additional_columns() -> None:
    inspector = inspect(engine)
    if not inspector.has_table("runs"):
        return

    existing_columns = {col["name"] for col in inspector.get_columns("runs")}
    statements = []

    if "universe_name" not in existing_columns:
        statements.append("ALTER TABLE runs ADD COLUMN universe_name VARCHAR")
    if "custom_symbols_blob" not in existing_columns:
        statements.append("ALTER TABLE runs ADD COLUMN custom_symbols_blob TEXT")
    if "market_mood_index" not in existing_columns:
        statements.append("ALTER TABLE runs ADD COLUMN market_mood_index FLOAT")
    if "market_mood_sentiment" not in existing_columns:
        statements.append("ALTER TABLE runs ADD COLUMN market_mood_sentiment VARCHAR")
    if "market_mood_description" not in existing_columns:
        statements.append("ALTER TABLE runs ADD COLUMN market_mood_description TEXT")
    if "market_mood_recommendation" not in existing_columns:
        statements.append("ALTER TABLE runs ADD COLUMN market_mood_recommendation TEXT")
    if "stock_evaluation_blob" not in existing_columns:
        statements.append("ALTER TABLE runs ADD COLUMN stock_evaluation_blob TEXT")

    if statements:
        with engine.begin() as connection:
            for stmt in statements:
                connection.execute(text(stmt))
    
    # Ensure additional columns for prediction_tracking table
    if not inspector.has_table("prediction_tracking"):
        return
    
    pred_columns = {col["name"] for col in inspector.get_columns("prediction_tracking")}
    pred_statements = []
    
    # Fundamental metrics at suggestion time
    for col_name in [
        "suggested_pe", "suggested_peg", "suggested_roe", "suggested_roic",
        "suggested_debt_to_equity", "suggested_interest_coverage",
        "suggested_revenue_growth", "suggested_profit_growth", "suggested_beta",
        # Current fundamental metrics
        "current_pe", "current_peg", "current_roe", "current_roic",
        "current_debt_to_equity", "current_interest_coverage",
        "current_revenue_growth", "current_profit_growth", "current_beta",
        "current_metrics_updated_at",
    ]:
        if col_name not in pred_columns:
            if col_name == "current_metrics_updated_at":
                pred_statements.append(f"ALTER TABLE prediction_tracking ADD COLUMN {col_name} DATETIME")
            else:
                pred_statements.append(f"ALTER TABLE prediction_tracking ADD COLUMN {col_name} FLOAT")
    
    if pred_statements:
        with engine.begin() as connection:
            for stmt in pred_statements:
                connection.execute(text(stmt))


def _serialize_snapshots(snapshots: List[StockSnapshot]) -> str:
    return json.dumps([snapshot.to_dict() for snapshot in snapshots], default=str)


def _serialize_news(news_map: Dict[str, List[NewsItem]]) -> str:
    payload = {
        symbol: [item.to_dict() for item in items] for symbol, items in news_map.items()
    }
    return json.dumps(payload, default=str)


def _serialize_fii_trend(fii_trend: Optional[pd.DataFrame]) -> Optional[str]:
    if fii_trend is None or fii_trend.empty:
        return None
    return fii_trend.to_json(orient="records", date_format="iso")


def _serialize_custom_symbols(symbols: Optional[List[str]]) -> Optional[str]:
    if not symbols:
        return None
    return json.dumps(symbols, default=str)


def _serialize_evaluation(evaluation: Optional[Dict[str, Any]]) -> Optional[str]:
    if not evaluation:
        return None
    return json.dumps(evaluation, default=str)


def _deserialize_snapshots(blob: str) -> List[StockSnapshot]:
    """Deserialize snapshots_blob back to StockSnapshot objects."""
    data_list = json.loads(blob)
    snapshots = []
    for item in data_list:
        # Recreate StockSnapshot - note: price_history and forecast are not stored
        # so we'll create a minimal snapshot for display purposes
        # If price_history is needed, it must be re-fetched
        snapshot = StockSnapshot(
            symbol=item.get("symbol", ""),
            ticker=item.get("ticker", item.get("symbol", "")),
            short_name=item.get("short_name"),
            price_history=pd.DataFrame(),  # Empty - will need to be re-fetched if needed
            change_1m=item.get("change_1m"),
            change_6m=item.get("change_6m"),
            fundamentals=item.get("fundamentals", {}),
            revenue_growth_yoy=item.get("revenue_growth_yoy"),
            revenue_cagr_3y=item.get("revenue_cagr_3y"),
            dividend_yield=item.get("dividend_yield"),
            free_cash_flow=item.get("free_cash_flow"),
            operating_margin=item.get("operating_margin"),
            promoter_holding_pct=item.get("promoter_holding_pct"),
            promoter_holding_change=item.get("promoter_holding_change"),
            forecast=None,  # Not stored
            forecast_slope=item.get("forecast_slope"),
            rsi_14=item.get("rsi_14"),
            moving_average_50=item.get("moving_average_50"),
            moving_average_200=item.get("moving_average_200"),
            macd=item.get("macd"),
            macd_signal=item.get("macd_signal"),
            bollinger_upper=item.get("bollinger_upper"),
            bollinger_lower=item.get("bollinger_lower"),
            bollinger_middle=item.get("bollinger_middle"),
            avg_volume_20=item.get("avg_volume_20"),
            volume_ratio=item.get("volume_ratio"),
            dist_52w_high=item.get("dist_52w_high"),
            dist_52w_low=item.get("dist_52w_low"),
        )
        snapshots.append(snapshot)
    return snapshots


def _deserialize_news(blob: str) -> Dict[str, List[NewsItem]]:
    """Deserialize news_blob back to news_map."""
    payload = json.loads(blob)
    news_map: Dict[str, List[NewsItem]] = {}
    for symbol, items in payload.items():
        news_items = []
        for item in items:
            published_at = None
            if item.get("published_at"):
                try:
                    # Try ISO format first
                    if isinstance(item["published_at"], str):
                        published_at = datetime.fromisoformat(item["published_at"].replace("Z", "+00:00"))
                    elif isinstance(item["published_at"], datetime):
                        published_at = item["published_at"]
                except Exception:
                    pass
            news_items.append(
                NewsItem(
                    symbol=item.get("symbol", symbol),
                    title=item.get("title", ""),
                    description=item.get("description"),
                    url=item.get("url", ""),
                    published_at=published_at,
                    source=item.get("source"),
                )
            )
        news_map[symbol] = news_items
    return news_map


def _deserialize_fii_trend(blob: Optional[str]) -> Optional[pd.DataFrame]:
    """Deserialize fii_blob back to DataFrame."""
    if not blob:
        return None
    try:
        return pd.read_json(blob, orient="records")
    except Exception:
        return None


def _deserialize_evaluation(blob: Optional[str]) -> Optional[Dict[str, Any]]:
    """Deserialize stock_evaluation_blob back to dict."""
    if not blob:
        return None
    try:
        return json.loads(blob)
    except Exception:
        return None


def log_run(
    horizon_years: int,
    stock_universe: int,
    invest_amount: float,
    strategy_notes: str,
    snapshots: List[StockSnapshot],
    news_map: Dict[str, List[NewsItem]],
    llm_result: LLMResult,
    llm_raw: Optional[str] = None,
    fii_trend: Optional[pd.DataFrame] = None,
    universe_name: Optional[str] = None,
    custom_symbols: Optional[List[str]] = None,
    market_mood: Optional[MarketMood] = None,
    stock_evaluation: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist a completed app run and return the run ID."""
    init_db()
    run_id = str(uuid.uuid4())
    evaluation_payload = stock_evaluation or llm_result.evaluation
    run_record = RunRecord(
        id=run_id,
        horizon_years=horizon_years,
        stock_universe=stock_universe,
        invest_amount=invest_amount,
        strategy_notes=strategy_notes or "",
        llm_summary=llm_result.summary,
        llm_guidance=llm_result.guidance,
        llm_raw=llm_raw,
        snapshots_blob=_serialize_snapshots(snapshots),
        news_blob=_serialize_news(news_map),
        fii_blob=_serialize_fii_trend(fii_trend),
        universe_name=universe_name,
        custom_symbols_blob=_serialize_custom_symbols(custom_symbols),
        market_mood_index=market_mood.index if market_mood else None,
        market_mood_sentiment=market_mood.sentiment if market_mood else None,
        market_mood_description=market_mood.description if market_mood else None,
        market_mood_recommendation=market_mood.recommendation if market_mood else None,
        stock_evaluation_blob=_serialize_evaluation(evaluation_payload),
    )

    for suggestion in llm_result.allocations:
        run_record.suggestions.append(
            SuggestionRecord(
                symbol=suggestion.symbol,
                allocation_pct=suggestion.allocation_pct,
                rationale=suggestion.rationale,
                risks=suggestion.risks,
            )
        )
        
        # Create prediction tracking record
        # Find the suggested price and fundamental metrics from snapshots
        suggested_price = None
        suggested_metrics = {}
        for snapshot in snapshots:
            if snapshot.symbol == suggestion.symbol:
                # Get latest price from price_history if available
                if snapshot.price_history is not None and not snapshot.price_history.empty:
                    if "Close" in snapshot.price_history.columns:
                        suggested_price = float(snapshot.price_history["Close"].iloc[-1])
                
                # Extract fundamental metrics at suggestion time
                suggested_metrics = {
                    "suggested_pe": snapshot.fundamentals.get("trailingPE"),
                    "suggested_peg": snapshot.peg_ratio,
                    "suggested_roe": snapshot.fundamentals.get("returnOnEquity"),
                    "suggested_roic": snapshot.roic,
                    "suggested_debt_to_equity": snapshot.fundamentals.get("debtToEquity"),
                    "suggested_interest_coverage": snapshot.interest_coverage,
                    "suggested_revenue_growth": snapshot.revenue_growth_yoy,
                    "suggested_profit_growth": snapshot.profit_growth_yoy,
                    "suggested_beta": snapshot.beta,
                }
                break
        
        if suggested_price is not None:
            run_record.predictions.append(
                PredictionTracking(
                    symbol=suggestion.symbol,
                    suggested_date=datetime.utcnow(),
                    suggested_price=suggested_price,
                    allocation_pct=suggestion.allocation_pct,
                    rationale=suggestion.rationale,
                    risks=suggestion.risks,
                    action_taken=None,  # User can update this later
                    **suggested_metrics,  # Include all fundamental metrics
                )
            )

    with SessionLocal() as session:
        session.add(run_record)
        session.commit()

    return run_id


def fetch_recent_runs(limit: int = 10) -> List[RunRecord]:
    init_db()
    with SessionLocal() as session:
        return (
            session.query(RunRecord)
            .options(selectinload(RunRecord.suggestions), selectinload(RunRecord.predictions))
            .order_by(RunRecord.created_at.desc())
            .limit(limit)
            .all()
        )


def fetch_run_by_id(run_id: str) -> Optional[RunRecord]:
    """Fetch a specific run by its ID."""
    init_db()
    with SessionLocal() as session:
        return (
            session.query(RunRecord)
            .options(selectinload(RunRecord.suggestions), selectinload(RunRecord.predictions))
            .filter(RunRecord.id == run_id)
            .first()
        )


def load_run_data(run: RunRecord) -> Dict[str, Any]:
    """Deserialize all stored data from a RunRecord."""
    return {
        "snapshots": _deserialize_snapshots(run.snapshots_blob),
        "news_map": _deserialize_news(run.news_blob),
        "fii_trend": _deserialize_fii_trend(run.fii_blob),
        "evaluation": _deserialize_evaluation(run.stock_evaluation_blob),
    }


def fetch_all_predictions(limit: Optional[int] = None) -> List[PredictionTracking]:
    """Fetch all prediction tracking records, optionally limited."""
    init_db()
    with SessionLocal() as session:
        query = session.query(PredictionTracking).order_by(
            PredictionTracking.suggested_date.desc()
        )
        if limit:
            query = query.limit(limit)
        return query.all()


def update_prediction_price(
    prediction_id: int, current_price: float, return_pct: Optional[float] = None
) -> bool:
    """Update the current price for a prediction tracking record."""
    init_db()
    with SessionLocal() as session:
        prediction = session.query(PredictionTracking).filter(
            PredictionTracking.id == prediction_id
        ).first()
        if not prediction:
            return False
        
        prediction.current_price = current_price
        prediction.current_price_updated_at = datetime.utcnow()
        
        # Calculate days since suggestion
        if prediction.suggested_date:
            delta = datetime.utcnow() - prediction.suggested_date
            prediction.days_since_suggestion = delta.days
        
        # Calculate return percentage if not provided
        if return_pct is None and prediction.suggested_price and prediction.suggested_price > 0:
            prediction.return_pct = (current_price - prediction.suggested_price) / prediction.suggested_price
        elif return_pct is not None:
            prediction.return_pct = return_pct
        
        session.commit()
        return True


def update_prediction_action(prediction_id: int, action_taken: str) -> bool:
    """Update the action_taken field for a prediction (bought/wait/skipped)."""
    init_db()
    with SessionLocal() as session:
        prediction = session.query(PredictionTracking).filter(
            PredictionTracking.id == prediction_id
        ).first()
        if not prediction:
            return False
        
        prediction.action_taken = action_taken
        session.commit()
        return True


def fetch_predictions_by_run_id(run_id: str) -> List[PredictionTracking]:
    """Fetch all predictions for a specific run."""
    init_db()
    with SessionLocal() as session:
        return (
            session.query(PredictionTracking)
            .filter(PredictionTracking.run_id == run_id)
            .order_by(PredictionTracking.suggested_date.desc())
            .all()
        )


def delete_unknown_runs() -> int:
    """Delete all runs where universe_name is None (displayed as 'Unknown').
    
    Returns:
        Number of runs deleted.
    """
    init_db()
    with SessionLocal() as session:
        # Find all runs with universe_name as None
        unknown_runs = session.query(RunRecord).filter(
            RunRecord.universe_name.is_(None)
        ).all()
        
        count = len(unknown_runs)
        
        # Delete these runs (cascade will handle suggestions and predictions)
        for run in unknown_runs:
            session.delete(run)
        
        session.commit()
        return count
