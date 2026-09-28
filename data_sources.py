"""
Market data sources. All return a DataFrame with columns
time (UTC, tz-aware), open, high, low, close, oldest first.
The LAST row is the bar still forming.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60}


class _Cached:
    min_refresh = 2.0

    def __init__(self):
        self._cache: dict[tuple, tuple[float, pd.DataFrame]] = {}

    def bars(self, tf: str, count: int) -> pd.DataFrame:
        key = (tf, count)
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < self.min_refresh:
            return hit[1]
        df = self._fetch(tf, count)
        self._cache[key] = (time.time(), df)
        return df

    def _fetch(self, tf, count):
        raise NotImplementedError

    def quote(self):
        return None


class MT5Source(_Cached):
    """Live Exness prices from the MetaTrader 5 terminal (Windows only)."""
    name = "Exness MT5"
    min_refresh = 2.0

    def __init__(self):
        super().__init__()
        import MetaTrader5 as mt5
        self.mt5 = mt5
        login = os.getenv("MT5_LOGIN")
        kwargs = {}
        if login:
            kwargs = dict(login=int(login), password=os.getenv("MT5_PASSWORD", ""),
                          server=os.getenv("MT5_SERVER", ""))
        if not mt5.initialize(**kwargs):
            raise RuntimeError(f"MT5 connection failed: {mt5.last_error()}. "
                               "Open the Exness MT5 terminal and check your login in .env")
        self.offset = pd.Timedelta(hours=float(os.getenv("MT5_SERVER_UTC_OFFSET", "0")))
        wanted = os.getenv("MT5_SYMBOL")
        names = [s.name for s in mt5.symbols_get() if s.name.upper().startswith("XAUUSD")]
        if wanted:
            self.symbol = wanted
        elif names:
            self.symbol = names[0]
        else:
            raise RuntimeError("No XAUUSD symbol on this MT5 account")
        mt5.symbol_select(self.symbol, True)
        self.tf_map = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
                       "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1}

    def _fetch(self, tf, count):
        rates = self.mt5.copy_rates_from_pos(self.symbol, self.tf_map[tf], 0, count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"MT5 returned no bars: {self.mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True) - self.offset
        return df[["time", "open", "high", "low", "close"]]

    def quote(self):
        t = self.mt5.symbol_info_tick(self.symbol)
        return None if t is None else dict(bid=float(t.bid), ask=float(t.ask))


class TwelveDataSource(_Cached):
    """Free Twelve Data API (8 requests/min, 800/day on the free plan)."""
    name = "Twelve Data"
    min_refresh = 110.0   # keeps one timeframe under 800 requests/day
    intervals = {"M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h"}

    def __init__(self, api_key: str):
        super().__init__()
        import requests
        self.requests = requests
        self.key = api_key
        self.symbol = "XAU/USD"

    def _fetch(self, tf, count):
        r = self.requests.get("https://api.twelvedata.com/time_series", timeout=15, params=dict(
            symbol=self.symbol, interval=self.intervals[tf], outputsize=min(count, 5000),
            timezone="UTC", apikey=self.key))
        data = r.json()
        if data.get("status") == "error" or "values" not in data:
            raise RuntimeError(f"Twelve Data error: {data.get('message', data)}")
        df = pd.DataFrame(data["values"])[::-1].reset_index(drop=True)
        df["time"] = pd.to_datetime(df["datetime"], utc=True)
        for k in ("open", "high", "low", "close"):
            df[k] = df[k].astype(float)
        return df[["time", "open", "high", "low", "close"]]


class DemoSource(_Cached):
    """Synthetic prices so the app can be tried without any account. NOT real data."""
    name = "Demo (synthetic data)"
    min_refresh = 1.0

    def __init__(self, seed: int = 7):
        super().__init__()
        self.symbol = "XAUUSD (demo)"
        self.rng = np.random.default_rng(seed)
        self.frames: dict[str, pd.DataFrame] = {}

    def _step(self, price, ts):
        ny_hour = (ts.tz_convert("America/New_York").hour)
        vol = 1.6 if 2 <= ny_hour < 5 or 7 <= ny_hour < 11 else 0.8
        return price * (1 + self.rng.normal(0, 0.00045 * vol))

    def _fetch(self, tf, count):
        m = TF_MINUTES[tf]
        now = pd.Timestamp.now(tz="UTC").floor(f"{m}min")
        df = self.frames.get(tf)
        if df is None:
            idx = pd.date_range(end=now, periods=count, freq=f"{m}min")
            price, rows = 3300.0, []
            for ts in idx:
                path = [price]
                for _ in range(4):
                    path.append(self._step(path[-1], ts))
                rows.append((ts, path[0], max(path), min(path), path[-1]))
                price = path[-1]
            df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
        while df["time"].iloc[-1] < now:
            ts = df["time"].iloc[-1] + pd.Timedelta(minutes=m)
            p0 = df["close"].iloc[-1]
            df.loc[len(df)] = [ts, p0, p0, p0, p0]
        last = df.index[-1]
        p = self._step(df.at[last, "close"], df.at[last, "time"])
        df.at[last, "close"] = p
        df.at[last, "high"] = max(df.at[last, "high"], p)
        df.at[last, "low"] = min(df.at[last, "low"], p)
        df = df.iloc[-count:].reset_index(drop=True)
        self.frames[tf] = df
        return df.copy()


def make_source():
    choice = os.getenv("DATA_SOURCE", "auto").lower()
    errors = []
    if choice in ("auto", "mt5"):
        try:
            return MT5Source()
        except Exception as e:  # noqa: BLE001
            errors.append(f"MT5: {e}")
            if choice == "mt5":
                raise
    if choice in ("auto", "twelvedata"):
        key = os.getenv("TWELVEDATA_API_KEY")
        if key:
            return TwelveDataSource(key)
        errors.append("Twelve Data: TWELVEDATA_API_KEY not set")
        if choice == "twelvedata":
            raise RuntimeError(errors[-1])
    print("Falling back to demo data. Reasons:", "; ".join(errors))
    return DemoSource()
