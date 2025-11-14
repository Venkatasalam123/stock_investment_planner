"""Helpers for fetching NSE-related datasets such as Nifty 500 and FII trends."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


NSE_BASE_URL = "https://www.nseindia.com"
NIFTY_500_ENDPOINT = (
    "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500"
)
FII_DII_ENDPOINT = (
    "https://www.nseindia.com/api/fiidiiTradeDisc?selectedType=FII"
)
NIFTY_500_CSV_FALLBACKS = [
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://www1.nseindia.com/content/indices/ind_nifty500list.csv",
]
FII_TREND_CSV_FALLBACKS = [
    "https://archives.nseindia.com/content/nsccl/fii_stats.csv",
    "https://www1.nseindia.com/content/nsccl/fii_stats.csv",
]
INDEX_CSV_MAP = {
    "NIFTY 50": [
        "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
        "https://www1.nseindia.com/content/indices/ind_nifty50list.csv",
    ],
    "NIFTY NEXT 50": [
        "https://archives.nseindia.com/content/indices/ind_niftynext50list.csv",
        "https://www1.nseindia.com/content/indices/ind_niftynext50list.csv",
    ],
    "NIFTY 100": [
        "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
        "https://www1.nseindia.com/content/indices/ind_nifty100list.csv",
    ],
    "NIFTY MIDCAP 150": [
        "https://archives.nseindia.com/content/indices/ind_niftymidcap150_list.csv",
        "https://www1.nseindia.com/content/indices/ind_niftymidcap150_list.csv",
    ],
    "NIFTY 200": [
        "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
        "https://www1.nseindia.com/content/indices/ind_nifty200list.csv",
    ],
    "NIFTY 500": NIFTY_500_CSV_FALLBACKS,
}
AVAILABLE_INDICES = list(INDEX_CSV_MAP.keys())

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/118.0.5993.117 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Origin": "https://www.nseindia.com",
}


class NSEAPIError(RuntimeError):
    """Raised when the NSE API does not return a successful payload."""


def _build_client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(30.0),
        headers=DEFAULT_HEADERS,
    )


def _prime_session(client: httpx.Client) -> None:
    response = client.get(NSE_BASE_URL)
    response.raise_for_status()


def _extract_json(response: httpx.Response) -> Dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise NSEAPIError("Failed to decode NSE response as JSON") from exc

    if not isinstance(payload, dict):
        raise NSEAPIError("Unexpected payload received from NSE")

    return payload


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((httpx.HTTPError, NSEAPIError)),
)
def fetch_nifty_500_constituents() -> pd.DataFrame:
    """Return the latest Nifty 500 constituents."""
    # Try the API endpoint first (tends to have richer metadata).
    try:
        with _build_client() as client:
            _prime_session(client)
            response = client.get(NIFTY_500_ENDPOINT)
            response.raise_for_status()
            payload = _extract_json(response)
        data = payload.get("data")
        if isinstance(data, list) and data:
            frame = pd.DataFrame(data)
            if "symbol" in frame.columns:
                frame.rename(columns={"meta": "metadata"}, inplace=True, errors="ignore")
                return frame
    except (httpx.HTTPError, NSEAPIError):
        pass

    # Fallback to CSV downloads when the JSON API is blocked (e.g., HTTP 403).
    last_error: Exception | None = None
    for url in NIFTY_500_CSV_FALLBACKS:
        try:
            with _build_client() as client:
                response = client.get(url)
                response.raise_for_status()
                csv_bytes = response.content
            frame = pd.read_csv(BytesIO(csv_bytes))
            if "Symbol" in frame.columns:
                frame.rename(columns=str.lower, inplace=True)
                frame.rename(columns={"symbol": "symbol"}, inplace=True)
            elif "symbol" not in frame.columns and "SYMBOL" in frame.columns:
                frame.rename(columns={"SYMBOL": "symbol"}, inplace=True)

            if "symbol" not in frame.columns:
                raise NSEAPIError(f"Nifty 500 CSV from {url} missing symbol column")

            return frame
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    raise NSEAPIError(
        "Unable to download Nifty 500 constituents via API or CSV fallbacks"
    ) from last_error


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((httpx.HTTPError, NSEAPIError)),
)
def fetch_fii_trend() -> pd.DataFrame:
    """Return recent Foreign Institutional Investor (FII) trend data."""
    try:
        with _build_client() as client:
            _prime_session(client)
            response = client.get(FII_DII_ENDPOINT)
            response.raise_for_status()
            payload = _extract_json(response)

        fii_data = payload.get("data")
        if isinstance(fii_data, list) and fii_data:
            frame = pd.DataFrame(fii_data)
            if "netBuySell" in frame.columns:
                frame["netBuySell"] = pd.to_numeric(frame["netBuySell"], errors="coerce")
            if "date" in frame.columns:
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
                frame.sort_values("date", inplace=True, ignore_index=True)
            return frame
    except (httpx.HTTPError, NSEAPIError):
        pass

    last_error: Exception | None = None
    for url in FII_TREND_CSV_FALLBACKS:
        try:
            with _build_client() as client:
                response = client.get(url)
                response.raise_for_status()
                csv_bytes = response.content
            frame = pd.read_csv(BytesIO(csv_bytes))
            frame.columns = [col.strip().lower().replace(" ", "_") for col in frame.columns]

            date_col = next((col for col in frame.columns if "date" in col), None)
            net_candidates = [col for col in frame.columns if "net" in col]
            net_col = net_candidates[0] if net_candidates else None

            if not date_col or not net_col:
                raise NSEAPIError(f"FII CSV at {url} missing required columns.")

            result = frame[[date_col, net_col]].copy()
            result.rename(columns={date_col: "date", net_col: "netBuySell"}, inplace=True)
            result["date"] = pd.to_datetime(result["date"], errors="coerce")
            result["netBuySell"] = pd.to_numeric(result["netBuySell"], errors="coerce")
            result.dropna(subset=["date"], inplace=True)
            result.sort_values("date", inplace=True, ignore_index=True)
            return result
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    # If all attempts fail, return an empty frame so callers can handle gracefully.
    return pd.DataFrame(columns=["date", "netBuySell"])


def top_symbols_from_nifty500(limit: int = 100) -> List[str]:
    """Convenience helper to get the first N symbols from the Nifty 500 index."""
    constituents = fetch_nifty_500_constituents()
    symbols = constituents.get("symbol", pd.Series(dtype=str)).dropna().tolist()
    return symbols[:limit]


def fetch_index_constituents(index_name: str) -> pd.DataFrame:
    """Fetch constituents for a given NSE index using published CSV lists."""
    normalized = index_name.strip().upper()
    if normalized == "NIFTY 500":
        return fetch_nifty_500_constituents()

    csv_urls = INDEX_CSV_MAP.get(normalized)
    if not csv_urls:
        raise NSEAPIError(f"Unsupported index name: {index_name}")

    last_error: Exception | None = None
    for url in csv_urls:
        try:
            with _build_client() as client:
                response = client.get(url)
                response.raise_for_status()
                csv_bytes = response.content
            frame = pd.read_csv(BytesIO(csv_bytes))
            frame.columns = [col.strip().lower() for col in frame.columns]
            if "symbol" not in frame.columns and "SYMBOL" in frame.columns:
                frame.rename(columns={"SYMBOL": "symbol"}, inplace=True)
            if "symbol" not in frame.columns:
                raise NSEAPIError(f"Index CSV from {url} missing symbol column")
            return frame
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    raise NSEAPIError(f"Unable to fetch constituents for index {index_name}") from last_error

