"""
ICT-style analysis and signal engine for XAUUSD.

Concepts implemented, all computed bar by bar with no look-ahead:
  - Swing highs/lows (fractal pivots, confirmed `swing_len` bars later)
  - Market structure: BOS (continuation) and CHoCH / MSS (reversal)
  - Liquidity pools: previous day high/low, Asian range high/low,
    equal highs/lows, and ordinary swing highs/lows
  - Liquidity sweeps (stop runs) of those pools
  - Fair value gaps (3-candle imbalance) and order blocks
  - Killzones (London, New York AM) in New York time
  - Premium/discount of the 24h dealing range, and the NY midnight open
  - Higher-timeframe bias from H1 (or H4) market structure

Signal model ("sweep -> market structure shift -> FVG retrace"):
  1. Price sweeps a liquidity pool (for example, takes out the Asian low).
  2. Within `sweep_window` bars it breaks structure the other way,
     leaving a fair value gap.
  3. A limit entry sits in that FVG, the stop goes beyond the sweep extreme,
     and the target is the next opposing liquidity pool (min RR enforced).

Every setup is tracked forward (fill, stop, target, expiry), so the same code
produces the live signals and the historical statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
MAJOR_LIQUIDITY = ("PDL", "PDH", "Asian Low", "Asian High", "EQL", "EQH")

# New York local time, [start, end) in hours
KILLZONES = (
    ("Asian", 20.0, 24.0),
    ("London", 2.0, 5.0),
    ("New York AM", 7.0, 10.0),
    ("London close", 10.0, 12.0),
)


@dataclass
class EngineConfig:
    swing_len: int = 3            # bars each side for a swing pivot
    htf_swing_len: int = 2
    atr_len: int = 14
    min_fvg_atr: float = 0.15     # ignore gaps smaller than this x ATR
    eq_tolerance_atr: float = 0.10
    sweep_window: int = 12        # max bars from sweep to structure shift
    fvg_wait: int = 3             # bars after the shift for an FVG to appear
    entry_mode: str = "ce"        # "ce" = 50% of the FVG, "edge" = near edge
    sl_buffer_atr: float = 0.15
    min_rr: float = 2.0
    max_rr: float = 8.0
    max_wait_bars: int = 12       # cancel the limit order after this many bars
    max_hold_bars: int = 48       # close the trade after this many bars
    spread: float = 0.30          # USD per ounce; set to your Exness spread
    killzone_only: bool = True
    signal_killzones: tuple = ("London", "New York AM")
    disp_body_atr: float = 1.0    # displacement candle body vs ATR


def killzone_at(ny_hour: float) -> str | None:
    for name, start, end in KILLZONES:
        if start <= ny_hour < end:
            return name
    return None


def _atr(h, l, c, n):
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def market_structure(h, l, c, L):
    """Swing pivots and structure breaks. Pivot at j is confirmed at j+L."""
    n = len(c)
    trend = np.zeros(n, dtype=int)
    state = 0
    sh = sl = None  # [swing_idx, price, broken]
    swing_highs, swing_lows, events = [], [], []
    for i in range(n):
        j = i - L
        if j >= L:
            if h[j] >= h[j - L:j + L + 1].max() and h[j] > h[j - L:j].max():
                sh = [j, h[j], False]
                swing_highs.append((j, float(h[j]), i))
            if l[j] <= l[j - L:j + L + 1].min() and l[j] < l[j - L:j].min():
                sl = [j, l[j], False]
                swing_lows.append((j, float(l[j]), i))
        if sh is not None and not sh[2] and c[i] > sh[1]:
            sh[2] = True
            events.append(dict(idx=i, dir=1, type="CHoCH" if state == -1 else "BOS",
                               level=float(sh[1]), swing_idx=sh[0]))
            state = 1
        if sl is not None and not sl[2] and c[i] < sl[1]:
            sl[2] = True
            events.append(dict(idx=i, dir=-1, type="CHoCH" if state == 1 else "BOS",
                               level=float(sl[1]), swing_idx=sl[0]))
            state = -1
        trend[i] = state
    return dict(trend=trend, swing_highs=swing_highs, swing_lows=swing_lows, events=events)


def _htf_trend(times, o, h, l, c, tf_minutes, cfg):
    rule = "4h" if tf_minutes >= 60 else "1h"
    frame = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=times)
    r = frame.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    if len(r) < 10:
        return np.zeros(len(c), dtype=int), rule
    ms = market_structure(r["high"].to_numpy(), r["low"].to_numpy(),
                          r["close"].to_numpy(), cfg.htf_swing_len)
    htf_close = (r.index + pd.Timedelta(rule)).to_numpy()
    bar_close = (times + pd.Timedelta(minutes=tf_minutes)).to_numpy()
    k = np.searchsorted(htf_close, bar_close, side="right") - 1
    trend = np.where(k >= 0, ms["trend"][np.clip(k, 0, None)], 0)
    return trend, rule


def analyze(df: pd.DataFrame, tf_minutes: int, cfg: EngineConfig | None = None,
            chart_bars: int = 300) -> dict:
    """Run the full analysis on CLOSED bars. df needs time (UTC), open, high, low, close."""
    cfg = cfg or EngineConfig()
    df = df.reset_index(drop=True)
    times = pd.DatetimeIndex(pd.to_datetime(df["time"], utc=True))
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    n = len(c)
    if n < 60:
        raise ValueError(f"Need at least 60 closed bars, got {n}")

    atr = _atr(h, l, c, cfg.atr_len)
    ny = times.tz_convert(NY)
    ny_hour = np.asarray(ny.hour) + np.asarray(ny.minute) / 60
    ny_date = np.asarray(ny.date)
    kz = [killzone_at(x) for x in ny_hour]
    htf, htf_rule = _htf_trend(times, o, h, l, c, tf_minutes, cfg)
    ms = market_structure(h, l, c, cfg.swing_len)
    bars_per_day = max(1, int(24 * 60 / tf_minutes))

    events_at: dict[int, list] = {}
    for e in ms["events"]:
        events_at.setdefault(e["idx"], []).append(e)
    sh_conf: dict[int, list] = {}
    for j, p, ci in ms["swing_highs"]:
        sh_conf.setdefault(ci, []).append((j, p))
    sl_conf: dict[int, list] = {}
    for j, p, ci in ms["swing_lows"]:
        sl_conf.setdefault(ci, []).append((j, p))

    levels: list[dict] = []
    sweeps: list[dict] = []
    fvgs_active: list[dict] = []
    fvgs_all: list[dict] = []
    obs: list[dict] = []
    setups: list[dict] = []
    last_sweep = {1: None, -1: None}
    pending = {1: None, -1: None}
    day_hi = day_lo = day_open = None
    asia_hi = asia_lo = None
    day_open_arr = np.full(n, np.nan)
    sp = cfg.spread

    def add_level(price, kind, side, i, origin):
        levels.append(dict(price=float(price), kind=kind, side=side, since=i,
                           origin=origin, active=True, swept_idx=None))

    def update_setups(i):
        for s in setups:
            if s["status"] not in ("pending", "active") or s["created"] >= i:
                continue
            d = s["dir"]
            if s["status"] == "pending":
                filled = (l[i] + sp <= s["entry"]) if d == 1 else (h[i] >= s["entry"])
                if filled:
                    s["status"], s["fill_idx"] = "active", i
                else:
                    tp_hit = (h[i] >= s["tp"]) if d == 1 else (l[i] + sp <= s["tp"])
                    if tp_hit:
                        s["status"], s["exit_idx"] = "missed", i
                    elif i - s["created"] >= cfg.max_wait_bars:
                        s["status"], s["exit_idx"] = "expired", i
                    continue
            # active trade: stop is checked first (conservative)
            stop = (l[i] <= s["sl"]) if d == 1 else (h[i] + sp >= s["sl"])
            tp_hit = (h[i] >= s["tp"]) if d == 1 else (l[i] + sp <= s["tp"])
            if stop:
                s.update(status="loss", exit_idx=i, exit_price=s["sl"], r=-1.0)
            elif i > s["fill_idx"] and tp_hit:
                s.update(status="win", exit_idx=i, exit_price=s["tp"], r=s["rr"])
            elif i - s["fill_idx"] >= cfg.max_hold_bars:
                px = c[i] if d == 1 else c[i] + sp
                s.update(status="timeout", exit_idx=i, exit_price=float(px),
                         r=float((px - s["entry"]) * d / s["risk"]))

    def make_setup(d, f, p, i):
        sw = p["sweep"]
        ext = l[sw["idx"]:i + 1].min() if d == 1 else h[sw["idx"]:i + 1].max()
        entry = (f["top"] + f["bottom"]) / 2 if cfg.entry_mode == "ce" else (
            f["top"] if d == 1 else f["bottom"])
        buf = cfg.sl_buffer_atr * atr[i]
        sl_px = ext - buf if d == 1 else ext + buf
        risk = (entry - sl_px) * d
        if risk <= 0.1 * atr[i]:
            return
        side = "high" if d == 1 else "low"
        targets = sorted((lv for lv in levels if lv["active"] and lv["side"] == side
                          and (lv["price"] - entry) * d > 0),
                         key=lambda lv: abs(lv["price"] - entry))
        tp, tp_kind = None, None
        for lv in targets:
            rr = (lv["price"] - entry) * d / risk
            if cfg.min_rr <= rr <= cfg.max_rr:
                tp, tp_kind = lv["price"], lv["kind"]
                break
        if tp is None:
            tp, tp_kind = entry + d * cfg.min_rr * risk, f"Fixed {cfg.min_rr:g}R"
        rr = (tp - entry) * d / risk

        lo = l[max(0, i - bars_per_day + 1):i + 1].min()
        hi = h[max(0, i - bars_per_day + 1):i + 1].max()
        eq = (hi + lo) / 2
        in_zone = entry < eq if d == 1 else entry > eq
        htf_ok, htf_against = htf[i] == d, htf[i] == -d
        open_ok = day_open is not None and ((entry < day_open) if d == 1 else (entry > day_open))
        kzn = kz[i]
        in_kz = kzn in cfg.signal_killzones
        major = sw["kind"] in MAJOR_LIQUIDITY
        if cfg.killzone_only and not in_kz:
            return
        score = (40 + 20 * htf_ok - 15 * htf_against + 15 * in_kz + 10 * major
                 + 10 * in_zone + 5 * p["disp"] + 5 * open_ok)
        score = int(max(5, min(95, score)))
        word = "buy" if d == 1 else "sell"
        zone_word = "discount" if d == 1 else "premium"
        htf_name = htf_rule.upper()
        reasons = [
            dict(ok=True, text=f"Swept {sw['kind']} liquidity at {sw['level']:.2f}"),
            dict(ok=True, text=f"{p['event']['type']} {'up' if d == 1 else 'down'} "
                               f"through {p['event']['level']:.2f}"),
            dict(ok=True, text=f"Entry in fair value gap {f['bottom']:.2f}-{f['top']:.2f}"),
            dict(ok=bool(htf_ok), text=f"{htf_name} structure "
                 + ("agrees" if htf_ok else "disagrees" if htf_against else "is neutral")),
            dict(ok=in_kz, text=f"Inside {kzn} killzone" if in_kz else "Outside London/NY killzones"),
            dict(ok=bool(in_zone), text=f"Entry in {zone_word} of 24h range"),
            dict(ok=bool(open_ok), text=f"Entry {'below' if d == 1 else 'above'} NY midnight open"),
            dict(ok=bool(p["disp"]), text="Strong displacement candle"),
        ]
        for s in setups:  # one live idea per direction
            if s["dir"] == d and s["status"] == "active":
                return
            if s["dir"] == d and s["status"] == "pending":
                s["status"], s["exit_idx"] = "replaced", i
        setups.append(dict(
            id=f"{'L' if d == 1 else 'S'}{i}", dir=d, word=word, created=i,
            entry=float(entry), sl=float(sl_px), tp=float(tp), tp_kind=tp_kind,
            rr=float(rr), risk=float(risk), score=score, reasons=reasons,
            status="pending", fill_idx=None, exit_idx=None, exit_price=None, r=None,
            sweep_kind=sw["kind"], killzone=kzn, htf_aligned=bool(htf_ok),
            htf_against=bool(htf_against), in_zone=bool(in_zone),
            fvg_top=float(f["top"]), fvg_bottom=float(f["bottom"])))

    for i in range(n):
        # ---- new New York day: previous day and Asian range become liquidity
        if i == 0 or ny_date[i] != ny_date[i - 1]:
            if i > 0 and day_hi is not None:
                for lv in levels:
                    if lv["active"] and lv["kind"] in MAJOR_LIQUIDITY[:4]:
                        lv["active"] = False
                add_level(day_hi, "PDH", "high", i, i)
                add_level(day_lo, "PDL", "low", i, i)
                if asia_hi is not None:
                    add_level(asia_hi, "Asian High", "high", i, i)
                    add_level(asia_lo, "Asian Low", "low", i, i)
            day_hi, day_lo, day_open = h[i], l[i], o[i]
            asia_hi = asia_lo = None
        else:
            day_hi, day_lo = max(day_hi, h[i]), min(day_lo, l[i])
        if ny_hour[i] >= 20:
            asia_hi = h[i] if asia_hi is None else max(asia_hi, h[i])
            asia_lo = l[i] if asia_lo is None else min(asia_lo, l[i])
        day_open_arr[i] = day_open

        # ---- track open setups against this bar
        update_setups(i)

        # ---- newly confirmed swings become liquidity (equal highs/lows merge)
        tol = cfg.eq_tolerance_atr * atr[i]
        for j, p in sh_conf.get(i, []):
            kind, price = "Swing High", p
            for lv in levels:
                if lv["active"] and lv["side"] == "high" and lv["kind"] in ("Swing High", "EQH") \
                        and abs(lv["price"] - p) <= tol:
                    lv["active"], kind, price = False, "EQH", max(p, lv["price"])
            add_level(price, kind, "high", i, j)
        for j, p in sl_conf.get(i, []):
            kind, price = "Swing Low", p
            for lv in levels:
                if lv["active"] and lv["side"] == "low" and lv["kind"] in ("Swing Low", "EQL") \
                        and abs(lv["price"] - p) <= tol:
                    lv["active"], kind, price = False, "EQL", min(p, lv["price"])
            add_level(price, kind, "low", i, j)
        swings = [lv for lv in levels if lv["active"] and lv["kind"].startswith("Swing")]
        for lv in swings[:-60]:
            lv["active"] = False
        if len(levels) > 400:
            levels[:] = [lv for lv in levels if lv["active"] or lv["swept_idx"] is not None][-300:]

        # ---- liquidity sweeps
        for d, side in ((1, "low"), (-1, "high")):
            hit = [lv for lv in levels if lv["active"] and lv["side"] == side and lv["since"] < i
                   and ((l[i] < lv["price"]) if side == "low" else (h[i] > lv["price"]))]
            if hit:
                for lv in hit:
                    lv["active"], lv["swept_idx"] = False, i
                hit.sort(key=lambda lv: (MAJOR_LIQUIDITY.index(lv["kind"])
                                         if lv["kind"] in MAJOR_LIQUIDITY else 99))
                sw = dict(dir=d, idx=i, kind=hit[0]["kind"], level=hit[0]["price"],
                          extreme=float(l[i] if d == 1 else h[i]))
                sweeps.append(sw)
                last_sweep[d] = sw
            elif last_sweep[d] is not None and i - last_sweep[d]["idx"] <= cfg.sweep_window:
                sw = last_sweep[d]
                sw["extreme"] = float(min(sw["extreme"], l[i]) if d == 1 else max(sw["extreme"], h[i]))

        # ---- fair value gaps: update, then detect
        still = []
        for f in fvgs_active:
            if f["dir"] == 1:
                dead = c[i] < f["bottom"] or l[i] <= f["bottom"]
            else:
                dead = c[i] > f["top"] or h[i] >= f["top"]
            if dead:
                f["active"], f["end_idx"] = False, i
            else:
                still.append(f)
        fvgs_active = still
        if i >= 2:
            if l[i] > h[i - 2] and l[i] - h[i - 2] >= cfg.min_fvg_atr * atr[i]:
                f = dict(dir=1, top=float(l[i]), bottom=float(h[i - 2]), idx=i - 1,
                         formed=i, active=True, end_idx=None)
                fvgs_active.append(f)
                fvgs_all.append(f)
            if h[i] < l[i - 2] and l[i - 2] - h[i] >= cfg.min_fvg_atr * atr[i]:
                f = dict(dir=-1, top=float(l[i - 2]), bottom=float(h[i]), idx=i - 1,
                         formed=i, active=True, end_idx=None)
                fvgs_active.append(f)
                fvgs_all.append(f)

        # ---- order blocks: invalidate on a close through
        for ob in obs:
            if ob["active"] and ((ob["dir"] == 1 and c[i] < ob["bottom"])
                                 or (ob["dir"] == -1 and c[i] > ob["top"])):
                ob["active"], ob["end_idx"] = False, i

        # ---- structure breaks: order blocks and sweep -> shift detection
        for e in events_at.get(i, []):
            d = e["dir"]
            for k in range(i - 1, max(-1, i - 16), -1):
                if (d == 1 and c[k] < o[k]) or (d == -1 and c[k] > o[k]):
                    obs.append(dict(dir=d, top=float(h[k]), bottom=float(l[k]), idx=k,
                                    created=i, active=True, end_idx=None))
                    break
            sw = last_sweep[d]
            if sw is not None and 0 < i - sw["idx"] <= cfg.sweep_window:
                disp = (abs(c[i] - o[i]) >= cfg.disp_body_atr * atr[i]
                        or abs(c[i] - sw["extreme"]) >= 2 * atr[i])
                pending[d] = dict(event=e, sweep=dict(sw), created=i, disp=bool(disp))
                last_sweep[d] = None

        # ---- a shift plus a fresh FVG becomes a setup
        for d in (1, -1):
            p = pending[d]
            if p is None:
                continue
            if i - p["created"] > cfg.fvg_wait:
                pending[d] = None
                continue
            ext = p["sweep"]["extreme"]
            cands = [f for f in fvgs_active if f["dir"] == d and p["sweep"]["idx"] < f["formed"] <= i
                     and ((f["bottom"] > ext) if d == 1 else (f["top"] < ext))]
            if cands:
                make_setup(d, cands[-1], p, i)
                pending[d] = None

    # ------------------------------------------------------------------ output
    start = max(0, n - chart_bars)
    ts = lambda i: int(times[i].timestamp())
    markers = []
    for s in setups:
        if s["created"] >= start:
            markers.append(dict(time=ts(s["created"]), kind="signal", dir=s["dir"],
                                text=f"{s['word'].title()} {s['score']}"))
    for sw in sweeps:
        if sw["idx"] >= start and sw["kind"] in MAJOR_LIQUIDITY:
            markers.append(dict(time=ts(sw["idx"]), kind="sweep", dir=sw["dir"],
                                text=f"{sw['kind']} swept"))
    for e in ms["events"]:
        if e["idx"] >= start:
            markers.append(dict(time=ts(e["idx"]), kind="structure", dir=e["dir"], text=e["type"]))

    zones = []
    for f in [f for f in fvgs_all if f["idx"] >= start and f["active"]][-8:]:
        zones.append(dict(kind="FVG", dir=f["dir"], top=f["top"], bottom=f["bottom"],
                          time=ts(f["idx"])))
    for ob in [ob for ob in obs if ob["idx"] >= start and ob["active"]][-4:]:
        zones.append(dict(kind="OB", dir=ob["dir"], top=ob["top"], bottom=ob["bottom"],
                          time=ts(ob["idx"])))

    last = n - 1
    px = c[last]
    active_levels = [lv for lv in levels if lv["active"]]
    above = sorted([lv for lv in active_levels if lv["price"] > px], key=lambda lv: lv["price"])
    below = sorted([lv for lv in active_levels if lv["price"] <= px], key=lambda lv: -lv["price"])

    def pick(lst):
        major = [lv for lv in lst if lv["kind"] in MAJOR_LIQUIDITY]
        minor = [lv for lv in lst if lv["kind"] not in MAJOR_LIQUIDITY][:3]
        return sorted(major + minor, key=lambda lv: abs(lv["price"] - px))[:6]

    out_levels = [dict(price=lv["price"], kind=lv["kind"], side=lv["side"])
                  for lv in pick(above) + pick(below)]

    def setup_out(s):
        return dict(
            id=s["id"], dir=s["dir"], word=s["word"], status=s["status"],
            time=ts(s["created"]), entry=round(s["entry"], 2), sl=round(s["sl"], 2),
            tp=round(s["tp"], 2), tp_kind=s["tp_kind"], rr=round(s["rr"], 2),
            risk=round(s["risk"], 2), score=s["score"], reasons=s["reasons"],
            killzone=s["killzone"], sweep_kind=s["sweep_kind"],
            fill_time=ts(s["fill_idx"]) if s["fill_idx"] is not None else None,
            exit_time=ts(s["exit_idx"]) if s["exit_idx"] is not None else None,
            r=None if s["r"] is None else round(s["r"], 2))

    open_setups = [s for s in setups if s["status"] in ("pending", "active")]
    current = setup_out(open_setups[-1]) if open_setups else None
    lo24 = l[max(0, last - bars_per_day + 1):].min()
    hi24 = h[max(0, last - bars_per_day + 1):].max()

    return dict(
        timeframe_minutes=tf_minutes,
        htf=htf_rule.upper(),
        markers=markers,
        zones=zones,
        levels=out_levels,
        day_open=None if np.isnan(day_open_arr[last]) else float(day_open_arr[last]),
        context=dict(
            htf_trend=int(htf[last]), ltf_trend=int(ms["trend"][last]),
            atr=round(float(atr[last]), 2), range_high=float(hi24), range_low=float(lo24),
            equilibrium=float((hi24 + lo24) / 2),
        ),
        current=current,
        history=[setup_out(s) for s in setups][-30:][::-1],
        stats=summarize(setups),
        setups_raw=setups,
    )


def summarize(setups: list[dict]) -> dict:
    done = [s for s in setups if s["status"] in ("win", "loss", "timeout")]
    if not done:
        return dict(trades=0, signals=len(setups))
    r = np.array([s["r"] for s in done])
    return dict(
        signals=len(setups),
        trades=len(done),
        wins=int(sum(s["status"] == "win" for s in done)),
        win_rate=round(float((r > 0).mean() * 100), 1),
        avg_r=round(float(r.mean()), 2),
        total_r=round(float(r.sum()), 2),
        not_filled=int(sum(s["status"] in ("missed", "expired", "replaced") for s in setups)),
    )
