"""News retrieval using Google News RSS feeds in parallel."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import feedparser
from urllib.parse import quote_plus
import re
import time

GOOGLE_NEWS_TEMPLATE = (
    "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
)
RSS_CACHE_TTL = 15 * 60  # 15 minutes
RSS_MAX_WORKERS = 8
_news_cache: Dict[str, Tuple[float, List["NewsItem"]]] = {}


class NewsServiceError(RuntimeError):
    """Raised when the news service fails."""


@dataclass
class NewsItem:
    symbol: str
    title: str
    description: str | None
    url: str
    published_at: Optional[datetime]
    source: str | None

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "symbol": self.symbol,
            "title": self.title,
            "description": self.description,
            "url": self.url,
            "published_at": (
                self.published_at.isoformat() if self.published_at else None
            ),
            "source": self.source,
        }


def _fetch_symbol_news(
    symbol: str,
    display_name: Optional[str],
    max_items: int,
) -> List[NewsItem]:
    cache_key = f"{symbol}:{max_items}"
    cached = _news_cache.get(cache_key)
    if cached:
        timestamp, items = cached
        if time.time() - timestamp <= RSS_CACHE_TTL:
            return items

    query_subject = display_name or symbol
    query = (
        f"{query_subject} stock site:moneycontrol.com OR site:reuters.com "
        f"OR site:economictimes.indiatimes.com"
    )
    feed_url = GOOGLE_NEWS_TEMPLATE.format(query=quote_plus(query))

    feed = feedparser.parse(feed_url)
    if feed.bozo:
        raise NewsServiceError(f"Failed to parse RSS feed for {symbol}")

    news: List[NewsItem] = []
    for entry in feed.entries[:max_items]:
        title = re.sub(r"<.*?>", "", entry.get("title", "").strip())
        link = entry.get("link", "")
        summary = entry.get("summary")
        if summary:
            summary = re.sub(r"<.*?>", "", summary)
        published_dt = None
        if entry.get("published_parsed"):
            published_dt = datetime.fromtimestamp(time.mktime(entry.published_parsed))
        source = None
        if "source" in entry and isinstance(entry.source, dict):
            source = entry.source.get("title")
        news.append(
            NewsItem(
                symbol=symbol,
                title=title,
                description=summary,
                url=link,
                published_at=published_dt,
                source=source,
            )
        )

    _news_cache[cache_key] = (time.time(), news)
    return news


def fetch_headlines(
    symbols: Iterable[str],
    api_key: Optional[str],  # unused but kept for compatibility
    max_per_symbol: int = 3,
    max_workers: int = RSS_MAX_WORKERS,
    symbol_to_name: Optional[Dict[str, str]] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Tuple[Dict[str, List[NewsItem]], Dict[str, str]]:
    """Fetch latest headlines per symbol using Google News RSS feeds."""
    headlines: Dict[str, List[NewsItem]] = {}
    errors: Dict[str, str] = {}
    symbol_to_name = symbol_to_name or {}

    symbols_list = list(dict.fromkeys(symbols))
    total = len(symbols_list)
    if total == 0:
        return headlines, errors

    workers = max(1, min(max_workers, total))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                _fetch_symbol_news,
                symbol,
                symbol_to_name.get(symbol),
                max_per_symbol,
            ): symbol
            for symbol in symbols_list
        }
        completed = 0
        for future in as_completed(future_map):
            symbol = future_map[future]
            completed += 1
            if progress_callback:
                progress_callback(symbol, completed, total)
            try:
                news_items = future.result()
            except Exception as exc:  # noqa: BLE001
                errors[symbol] = str(exc)
            else:
                if news_items:
                    headlines[symbol] = news_items

    return headlines, errors

