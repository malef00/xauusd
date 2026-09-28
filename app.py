"""
XAUUSD ICT signal desk - local web app.

Run:  python app.py   then open http://127.0.0.1:5000
"""
from __future__ import annotations

import os
import time

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

from data_sources import TF_MINUTES, make_source  # noqa: E402
from ict_engine import NY, EngineConfig, analyze, killzone_at  # noqa: E402

HISTORY_BARS = int(os.getenv("HISTORY_BARS", "3000"))
CFG = EngineConfig(
    spread=float(os.getenv("SPREAD_USD", "0.30")),
    min_rr=float(os.getenv("MIN_RR", "2.0")),
    killzone_only=os.getenv("KILLZONE_ONLY", "1") == "1",
)

app = Flask(__name__)
source = make_source()
_analysis: dict[str, tuple] = {}


def _session_info(now_utc: pd.Timestamp) -> dict:
    ny = now_utc.tz_convert(NY)
    hour = ny.hour + ny.minute / 60
    return dict(killzone=killzone_at(hour), ny_time=ny.strftime("%H:%M"),
                weekend=ny.weekday() == 5 or (ny.weekday() == 6 and hour < 18)
                or (ny.weekday() == 4 and hour >= 17))


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def state():
    tf = request.args.get("tf", "M5").upper()
    if tf not in TF_MINUTES:
        return jsonify(error=f"Unknown timeframe {tf}"), 400
    try:
        df = source.bars(tf, HISTORY_BARS)
    except Exception as e:  # noqa: BLE001
        return jsonify(error=str(e), source=source.name), 503

    closed = df.iloc[:-1]
    key = (str(closed["time"].iloc[-1]), len(closed), CFG.spread)
    cached = _analysis.get(tf)
    if cached is None or cached[0] != key:
        t0 = time.time()
        result = analyze(closed, TF_MINUTES[tf], CFG)
        result.pop("setups_raw")
        result["compute_ms"] = int((time.time() - t0) * 1000)
        _analysis[tf] = (key, result)
    result = dict(_analysis[tf][1])

    live = df.iloc[-1]
    quote = source.quote()
    price = quote["bid"] if quote else float(live["close"])
    candles = [dict(time=int(r.time.timestamp()), open=float(r.open), high=float(r.high),
                    low=float(r.low), close=float(r.close)) for r in df.iloc[-300:].itertuples()]

    cur = result.get("current")
    if cur:
        cur = dict(cur)
        d = cur["dir"]
        cur["distance"] = round((price - cur["entry"]) * d, 2)
        cur["live_r"] = round((price - cur["entry"]) * d / cur["risk"], 2)
        if cur["status"] == "pending" and (price - cur["entry"]) * d <= 0:
            cur["note"] = "Price is at the entry now"
        result["current"] = cur

    result.update(
        source=source.name, symbol=source.symbol, timeframe=tf, candles=candles,
        price=price, quote=quote, poll_seconds=max(2, int(getattr(source, "min_refresh", 5))),
        session=_session_info(pd.Timestamp.now(tz="UTC")),
        server_time=int(time.time()), spread_setting=CFG.spread,
    )
    return jsonify(result)


if __name__ == "__main__":
    print(f"Data source: {source.name} ({source.symbol})")
    port = int(os.getenv("PORT", "5000"))
    print(f"Open http://127.0.0.1:{port} in your browser")
    if os.getenv("OPEN_BROWSER", "1") == "1":
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(host="127.0.0.1", port=port, debug=False)
