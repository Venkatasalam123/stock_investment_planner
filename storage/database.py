"""SQLite persistence helpers for tracking Streamlit app executions."""

from __future__ import annotations

import json
import os
import pathlib
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urlparse, urlunparse

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import DateTime, Float, ForeignKey, String, Text, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy import inspect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker, selectinload

from services.core.market_data import StockSnapshot
from services.core.market_mood import MarketMood
from services.core.news import NewsItem
from services.core.llm import LLMResult

# Ensure .env is loaded even when this module is imported directly
load_dotenv()

def _mask_password_in_url(url: str) -> str:
    """Mask password in URL for safe logging."""
    if not url or url.startswith("sqlite"):
        return url
    
    try:
        parsed = urlparse(url)
        if parsed.password:
            masked = f"{parsed.scheme}://{parsed.username}:***@{parsed.hostname}"
            if parsed.port:
                masked += f":{parsed.port}"
            masked += parsed.path
            if parsed.query:
                masked += f"?{parsed.query}"
            return masked
    except Exception:
        pass
    
    return url


def _print_safe_url(label: str, url: str) -> None:
    """Print database URL with masked password."""
    safe_url = _mask_password_in_url(url)
    print(f"{label}: {safe_url}")


def _debug_connection_info(app_env: str, local_url: str | None, st_secrets: dict | None) -> None:
    """Print debug information about connection resolution."""
    print("=" * 60)
    print("🔍 DATABASE CONNECTION DEBUG")
    print("=" * 60)
    print(f"APP_ENV from os.getenv: {os.getenv('APP_ENV')}")
    print(f"ENV from os.getenv: {os.getenv('ENV')}")
    print(f"APP_ENV resolved: {app_env}")
    print(f"LOCAL_DATABASE_URL from os.getenv: {os.getenv('LOCAL_DATABASE_URL')}")
    print(f"DATABASE_URL from os.getenv: {'SET' if os.getenv('DATABASE_URL') else 'NOT SET'}")
    
    if st_secrets:
        print(f"st.secrets available: YES")
        print(f"APP_ENV from st.secrets: {st_secrets.get('APP_ENV', 'NOT SET')}")
        print(f"LOCAL_DATABASE_URL from st.secrets: {'SET' if st_secrets.get('LOCAL_DATABASE_URL') else 'NOT SET'}")
        print(f"DATABASE_URL from st.secrets: {'SET' if st_secrets.get('DATABASE_URL') else 'NOT SET'}")
    else:
        print(f"st.secrets available: NO (not in Streamlit context)")
    
    print("=" * 60)


def _normalize_database_url(url: str) -> str:
    """Normalize database URL, ensuring password is properly encoded."""
    if not url or url.startswith("sqlite"):
        return url
    
    try:
        parsed = urlparse(url)
        if parsed.password:
            encoded_password = quote_plus(parsed.password)
            netloc = f"{parsed.username}:{encoded_password}@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            normalized = urlunparse((
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment
            ))
            return normalized
    except Exception:
        pass
    
    return url


def _resolve_database_url() -> str:
    """
    Decide which database URL to use.
    - Local/default: sqlite (or LOCAL_DATABASE_URL if provided)
    - Cloud/production: DATABASE_URL (Neon/Postgres)
    
    Use APP_ENV (or ENV) to detect cloud deployments.
    Also checks Streamlit secrets if available.
    """
    # Try to import streamlit to check secrets (only in Streamlit context)
    st_secrets = None
    try:
        import streamlit as st
        try:
            st_secrets = st.secrets.to_dict()
        except Exception:
            pass  # st.secrets not available (not in Streamlit context)
    except ImportError:
        pass  # streamlit not installed (shouldn't happen but safe)
    
    # Check environment variables and Streamlit secrets
    app_env = (os.getenv("APP_ENV") or os.getenv("ENV") or "").lower()
    if not app_env and st_secrets:
        app_env = str(st_secrets.get("APP_ENV", "")).lower()
    
    local_url = os.getenv("LOCAL_DATABASE_URL")
    if not local_url and st_secrets:
        local_url = st_secrets.get("LOCAL_DATABASE_URL")
    
    default_sqlite = "sqlite:///data/app.db"
    
    # Debug: Print connection info (masking password)
    _debug_connection_info(app_env, local_url, st_secrets)
    
    if app_env in {"cloud", "prod", "production", "streamlit"}:
        cloud_url = os.getenv("DATABASE_URL")
        if not cloud_url and st_secrets:
            cloud_url = st_secrets.get("DATABASE_URL")
        
        if not cloud_url:
            error_msg = f"DATABASE_URL must be provided when APP_ENV is '{app_env}'. "
            error_msg += f"Checked os.getenv('DATABASE_URL') and st.secrets.get('DATABASE_URL')."
            print(f"❌ DATABASE ERROR: {error_msg}")
            raise RuntimeError(error_msg)
        
        normalized_url = _normalize_database_url(cloud_url)
        _print_safe_url("✅ Using CLOUD database (Neon/Postgres)", normalized_url)
        return normalized_url
    
    # Local/dev: prefer LOCAL_DATABASE_URL, else DATABASE_URL, else sqlite
    if local_url:
        normalized_url = _normalize_database_url(local_url)
        _print_safe_url("✅ Using LOCAL_DATABASE_URL", normalized_url)
        return normalized_url
    
    fallback = os.getenv("DATABASE_URL")
    if not fallback and st_secrets:
        fallback = st_secrets.get("DATABASE_URL")
    fallback = fallback or default_sqlite
    
    normalized_url = _normalize_database_url(fallback)
    if fallback.startswith("sqlite"):
        print(f"✅ Using SQLite database: {fallback}")
    else:
        _print_safe_url("✅ Using fallback DATABASE_URL", normalized_url)
    return normalized_url


DATABASE_URL = _resolve_database_url()


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
    horizon_months: Mapped[int] = mapped_column(nullable=True)  # New field for months-based horizon
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


def get_connection_info() -> dict[str, str]:
    """Get connection information for UI display (safe - no passwords)."""
    info: dict[str, str] = {}
    
    # Try to get from environment and Streamlit secrets
    try:
        import streamlit as st
        try:
            st_secrets = st.secrets.to_dict()
        except Exception:
            st_secrets = None
    except (ImportError, RuntimeError):
        st_secrets = None
    
    # Get APP_ENV
    app_env = os.getenv("APP_ENV") or os.getenv("ENV") or ""
    if not app_env and st_secrets:
        app_env = str(st_secrets.get("APP_ENV", ""))
    info["Environment"] = app_env if app_env else "Not set (defaults to LOCAL)"
    
    # Get database type and URL
    db_url = DATABASE_URL
    if db_url.startswith("sqlite"):
        info["Database Type"] = "SQLite (Local)"
        info["Database Path"] = db_url.replace("sqlite:///", "")
    elif "postgresql" in db_url.lower() or "postgres" in db_url.lower():
        info["Database Type"] = "PostgreSQL (Neon/Cloud)"
        info["Database URL"] = _mask_password_in_url(db_url)
    else:
        info["Database Type"] = "Unknown"
        info["Database URL"] = _mask_password_in_url(db_url)
    
    # Source of configuration
    sources = []
    if os.getenv("DATABASE_URL") or os.getenv("LOCAL_DATABASE_URL") or os.getenv("APP_ENV"):
        sources.append("Environment Variables (.env)")
    if st_secrets:
        sources.append("Streamlit Secrets")
    if not sources:
        sources.append("Default (SQLite)")
    info["Configuration Source"] = ", ".join(sources)
    
    return info


def test_db_connection() -> tuple[bool, str]:
    """Test database connection and return (success, message)."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_type = "PostgreSQL (Neon)" if "postgresql" in DATABASE_URL.lower() else "SQLite"
        # Mask password in URL for display
        masked_url = DATABASE_URL
        if "@" in masked_url and ":" in masked_url.split("@")[0]:
            user_pass, rest = masked_url.split("@", 1)
            if ":" in user_pass:
                user, _ = user_pass.split(":", 1)
                masked_url = f"{user}:***@{rest}"
        return True, f"✅ Connected to {db_type}: {masked_url}"
    except Exception as exc:
        return False, f"❌ Connection failed: {str(exc)}"


def init_db() -> None:
    """Initialize database tables."""
    try:
        Base.metadata.create_all(engine)
        _ensure_additional_columns()
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize database: {exc}") from exc


def _ensure_additional_columns() -> None:
    """Add missing columns to existing tables (for migrations)."""
    inspector = inspect(engine)
    if not inspector.has_table("runs"):
        return

    existing_columns = {col["name"] for col in inspector.get_columns("runs")}
    statements = []
    
    # Use database-agnostic types (SQLAlchemy will translate)
    is_postgres = "postgresql" in DATABASE_URL.lower()

    if "universe_name" not in existing_columns:
        col_type = "VARCHAR" if not is_postgres else "VARCHAR(255)"
        statements.append(f"ALTER TABLE runs ADD COLUMN universe_name {col_type}")
    if "custom_symbols_blob" not in existing_columns:
        statements.append("ALTER TABLE runs ADD COLUMN custom_symbols_blob TEXT")
    if "market_mood_index" not in existing_columns:
        col_type = "FLOAT" if not is_postgres else "DOUBLE PRECISION"
        statements.append(f"ALTER TABLE runs ADD COLUMN market_mood_index {col_type}")
    if "market_mood_sentiment" not in existing_columns:
        col_type = "VARCHAR" if not is_postgres else "VARCHAR(50)"
        statements.append(f"ALTER TABLE runs ADD COLUMN market_mood_sentiment {col_type}")
    if "market_mood_description" not in existing_columns:
        statements.append("ALTER TABLE runs ADD COLUMN market_mood_description TEXT")
    if "market_mood_recommendation" not in existing_columns:
        statements.append("ALTER TABLE runs ADD COLUMN market_mood_recommendation TEXT")
    if "stock_evaluation_blob" not in existing_columns:
        statements.append("ALTER TABLE runs ADD COLUMN stock_evaluation_blob TEXT")
    if "horizon_months" not in existing_columns:
        col_type = "INTEGER" if not is_postgres else "INTEGER"
        statements.append(f"ALTER TABLE runs ADD COLUMN horizon_months {col_type}")
        # Migrate existing data: set horizon_months = horizon_years * 12
        if "horizon_years" in existing_columns:
            statements.append("UPDATE runs SET horizon_months = horizon_years * 12 WHERE horizon_months IS NULL")

    if statements:
        try:
            with engine.begin() as connection:
                for stmt in statements:
                    connection.execute(text(stmt))
        except Exception as exc:
            # Log but don't fail - columns might already exist or be incompatible
            print(f"Warning: Could not add some columns: {exc}")
    
    # Ensure additional columns for prediction_tracking table
    if not inspector.has_table("prediction_tracking"):
        return
    
    pred_columns = {col["name"] for col in inspector.get_columns("prediction_tracking")}
    pred_statements = []
    
    # Use database-agnostic types
    is_postgres = "postgresql" in DATABASE_URL.lower()
    
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
                col_type = "DATETIME" if not is_postgres else "TIMESTAMP"
                pred_statements.append(f"ALTER TABLE prediction_tracking ADD COLUMN {col_name} {col_type}")
            else:
                col_type = "FLOAT" if not is_postgres else "DOUBLE PRECISION"
                pred_statements.append(f"ALTER TABLE prediction_tracking ADD COLUMN {col_name} {col_type}")
    
    if pred_statements:
        try:
            with engine.begin() as connection:
                for stmt in pred_statements:
                    connection.execute(text(stmt))
        except Exception as exc:
            # Log but don't fail - columns might already exist or be incompatible
            print(f"Warning: Could not add some prediction_tracking columns: {exc}")


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


def _safe_float(value: Any) -> Optional[float]:
    """Convert numeric values (including numpy types) to Python floats."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
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
    horizon_months: Optional[int] = None,
) -> str:
    """Persist a completed app run and return the run ID."""
    init_db()
    run_id = str(uuid.uuid4())
    evaluation_payload = stock_evaluation or llm_result.evaluation
    # Compute horizon_months if not provided (backward compatibility)
    if horizon_months is None:
        horizon_months = int(horizon_years * 12)
    run_record = RunRecord(
        id=run_id,
        horizon_years=horizon_years,
        horizon_months=horizon_months,
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
        market_mood_index=_safe_float(market_mood.index) if market_mood else None,
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
                        suggested_price = _safe_float(snapshot.price_history["Close"].iloc[-1])
                
                # Extract fundamental metrics at suggestion time
                suggested_metrics = {
                    "suggested_pe": _safe_float(snapshot.fundamentals.get("trailingPE")),
                    "suggested_peg": _safe_float(snapshot.peg_ratio),
                    "suggested_roe": _safe_float(snapshot.fundamentals.get("returnOnEquity")),
                    "suggested_roic": _safe_float(snapshot.roic),
                    "suggested_debt_to_equity": _safe_float(snapshot.fundamentals.get("debtToEquity")),
                    "suggested_interest_coverage": _safe_float(snapshot.interest_coverage),
                    "suggested_revenue_growth": _safe_float(snapshot.revenue_growth_yoy),
                    "suggested_profit_growth": _safe_float(snapshot.profit_growth_yoy),
                    "suggested_beta": _safe_float(snapshot.beta),
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
        # Verify the record was saved
        saved = session.query(RunRecord).filter(RunRecord.id == run_id).first()
        if saved:
            print(f"✅ Run {run_id[:8]}... saved successfully to database")
        else:
            print(f"⚠️ Warning: Run {run_id[:8]}... commit succeeded but record not found immediately")

    return run_id


def fetch_recent_runs(limit: int = 10) -> List[RunRecord]:
    init_db()
    with SessionLocal() as session:
        # First, check total count
        total_count = session.query(RunRecord).count()
        print(f"📊 Total runs in database: {total_count}")
        
        runs = (
            session.query(RunRecord)
            .options(selectinload(RunRecord.suggestions), selectinload(RunRecord.predictions))
            .order_by(RunRecord.created_at.desc())
            .limit(limit)
            .all()
        )
        
        print(f"📊 Fetched {len(runs)} recent run(s) (limit={limit})")
        if runs:
            for run in runs[:3]:  # Print first 3 for debugging
                print(f"  - Run {run.id[:8]}... created at {run.created_at}, universe: {run.universe_name}")
        
        return runs


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
    prediction_id: int,
    current_price: float,
    return_pct: Optional[float] = None,
    current_metrics: Optional[Dict[str, Any]] = None,
) -> bool:
    """Update the current price (and optionally current fundamentals) for a prediction tracking record."""
    init_db()
    with SessionLocal() as session:
        prediction = session.query(PredictionTracking).filter(
            PredictionTracking.id == prediction_id
        ).first()
        if not prediction:
            return False
        
        now = datetime.utcnow()
        prediction.current_price = current_price
        prediction.current_price_updated_at = now
        
        # Calculate days since suggestion
        if prediction.suggested_date:
            delta = datetime.utcnow() - prediction.suggested_date
            prediction.days_since_suggestion = delta.days
        
        # Calculate return percentage if not provided
        if return_pct is None and prediction.suggested_price and prediction.suggested_price > 0:
            prediction.return_pct = (current_price - prediction.suggested_price) / prediction.suggested_price
        elif return_pct is not None:
            prediction.return_pct = return_pct

        # Optionally update current fundamental metrics snapshot
        if current_metrics:
            prediction.current_pe = current_metrics.get("pe")
            prediction.current_peg = current_metrics.get("peg")
            prediction.current_roe = current_metrics.get("roe")
            prediction.current_roic = current_metrics.get("roic")
            prediction.current_debt_to_equity = current_metrics.get("debt_to_equity")
            prediction.current_interest_coverage = current_metrics.get("interest_coverage")
            prediction.current_revenue_growth = current_metrics.get("revenue_growth")
            prediction.current_profit_growth = current_metrics.get("profit_growth")
            prediction.current_beta = current_metrics.get("beta")
            prediction.current_metrics_updated_at = now
        
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
    """Deprecated: kept for backward compatibility, returns 0 and performs no action."""
    return 0
