"""
Backtest the signal engine on historical XAUUSD bars.

Examples:
    python backtest.py --source mt5 --tf M5 --bars 50000
    python backtest.py --csv XAUUSDm_history.csv --tf M1
    python backtest.py --source demo --tf M5            (synthetic, for testing only)

It prints results for every setup, then broken down by each confluence,
so you can see which ICT conditions actually help on real gold data.
Trades are saved to backtest_trades.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from data_sources import TF_MINUTES, DemoSource, MT5Source, TwelveDataSource
from ict_engine import EngineConfig, analyze


def stats(rows: list[dict]) -> str:
    done = [s for s in rows if s["status"] in ("win", "loss", "timeout")]
    if not done:
        return "no completed trades"
    r = np.array([s["r"] for s in done])
    return (f"{len(done):4d} trades | win {100 * (r > 0).mean():5.1f}% | "
            f"avg {r.mean():+.2f}R | total {r.sum():+7.1f}R")


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["mt5", "twelvedata", "demo"], default=None)
    ap.add_argument("--csv", help="CSV with time,open,high,low,close (UTC or server time)")
    ap.add_argument("--tf", default="M5", choices=list(TF_MINUTES))
    ap.add_argument("--bars", type=int, default=50000)
    ap.add_argument("--spread", type=float, default=0.30)
    ap.add_argument("--server-utc-offset", type=float, default=0.0,
                    help="hours to subtract from CSV times to get UTC")
    ap.add_argument("--all-sessions", action="store_true",
                    help="include setups outside London/NY killzones")
    a = ap.parse_args()

    if a.csv:
        df = pd.read_csv(a.csv)
        df["time"] = pd.to_datetime(df["time"], utc=True) - pd.Timedelta(hours=a.server_utc_offset)
        df = df[["time", "open", "high", "low", "close"]].tail(a.bars)
    else:
        import os
        src = {"mt5": MT5Source, "demo": DemoSource,
               "twelvedata": lambda: TwelveDataSource(os.environ["TWELVEDATA_API_KEY"])
               }[a.source or "demo"]()
        df = src.bars(a.tf, a.bars).iloc[:-1]

    cfg = EngineConfig(spread=a.spread, killzone_only=not a.all_sessions)
    print(f"Bars: {len(df):,}  {df['time'].iloc[0]}  ->  {df['time'].iloc[-1]}")
    res = analyze(df, TF_MINUTES[a.tf], cfg)
    S = res["setups_raw"]

    print("\nAll setups:           ", stats(S))
    print("  never filled:       ", sum(s["status"] in ("missed", "expired", "replaced") for s in S))
    print("\nBy confluence")
    print("  HTF agrees          ", stats([s for s in S if s["htf_aligned"]]))
    print("  HTF disagrees       ", stats([s for s in S if s["htf_against"]]))
    print("  Discount/premium ok ", stats([s for s in S if s["in_zone"]]))
    print("  Wrong half of range ", stats([s for s in S if not s["in_zone"]]))
    for kz in sorted({str(s["killzone"]) for s in S}):
        print(f"  Killzone {kz:<11}", stats([s for s in S if str(s["killzone"]) == kz]))
    for kind in sorted({s["sweep_kind"] for s in S}):
        print(f"  Swept {kind:<14}", stats([s for s in S if s["sweep_kind"] == kind]))
    print("\nBy confluence score")
    for lo, hi in ((0, 50), (50, 65), (65, 80), (80, 101)):
        print(f"  {lo:2d}-{min(hi, 100):3d}             ", stats([s for s in S if lo <= s["score"] < hi]))

    t = pd.DataFrame([{k: v for k, v in s.items() if k != "reasons"} for s in S])
    if len(t):
        times = pd.to_datetime(df["time"].reset_index(drop=True), utc=True)
        t["created_time"] = times.iloc[t["created"]].to_numpy()
        t.to_csv("backtest_trades.csv", index=False)
        print("\nSaved backtest_trades.csv")
    print("\nReminder: past results do not guarantee future results. "
          "Beware of tuning settings until the numbers look good (overfitting).")


if __name__ == "__main__":
    main()
