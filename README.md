---
title: XAUUSD Signal Desk
emoji: 🪙
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# XAUUSD ICT signal desk

A local web app that reads live gold prices from your Exness MT5 terminal
(or the free Twelve Data API), maps ICT structure on the chart, and raises
day-trading signals with entry, stop and target.

## Quick start (Windows)

1. Install Python 3.10+ from python.org and tick "Add Python to PATH".
2. Install the Exness MT5 terminal, log in to a **demo** account, and leave it open.
3. Copy `.env.example` to `.env`. Your MT5 login can stay blank if the terminal
   is already logged in. Set `SPREAD_USD` to your account's typical gold spread.
4. Double-click `start.bat`. It installs everything and opens
   http://127.0.0.1:5000.

No MT5? Get a free key at twelvedata.com and put it in `TWELVEDATA_API_KEY`.
The free plan allows about 800 requests a day, so the app refreshes roughly
every 2 minutes in that mode. With no data source configured, the app runs on
clearly labelled synthetic demo data.

Manual start: `pip install -r requirements.txt` then `python app.py`.

## What the app looks for

| Concept | How it's detected |
|---|---|
| Swing highs/lows | Fractal pivot, 3 bars each side, confirmed 3 bars later |
| BOS / CHoCH | Candle close through the last swing high/low |
| Liquidity | Previous day high/low, Asian range (8pm-midnight NY), equal highs/lows, swings |
| Sweep | Price trades through a liquidity level |
| Fair value gap | 3-candle imbalance at least 0.15 x ATR |
| Order block | Last opposite candle before a structure break |
| Killzones | London 2-5am, New York 7-10am (New York time) |
| Bias | H1 market structure, 24h premium/discount, NY midnight open |

**Signal model:** liquidity sweep, then a structure shift the other way within
12 bars, then a new FVG. Entry is a limit order at the FVG midpoint, the stop
goes beyond the sweep extreme, and the target is the next opposing liquidity
pool with at least 2R. By default signals only fire in the London and New York
AM killzones. The confluence score (0-100) adds points for HTF agreement,
killzone, a major liquidity sweep, premium/discount, the midnight open and
displacement. It is a checklist, not a probability.

Everything is computed on closed candles only, so signals do not repaint.

## Test before trusting it

```
python backtest.py --source mt5 --tf M5 --bars 50000
python backtest.py --csv XAUUSDm_history.csv --tf M1
python backtest.py --source mt5 --tf M5 --all-sessions
```

The backtest uses the exact same engine, includes your spread, assumes the
stop is hit first when a candle touches both stop and target, and prints
results by each confluence and by score bucket. Use that breakdown to decide
which conditions to require, for example only taking trades with HTF agreement.
Change one setting at a time, and confirm any improvement on a period you did
not tune on.

## Put it online for free

Online hosts run Linux, so they use Twelve Data prices, not MT5.
Get a free key at twelvedata.com first.

**Render (link like `your-app.onrender.com`)**
1. Upload this folder to a new GitHub repository.
2. On render.com, choose New > Blueprint and pick the repository. It reads `render.yaml`.
3. When asked, paste your `TWELVEDATA_API_KEY`, then deploy.

**Hugging Face Spaces (link like `your-name-xauusd.hf.space`)**
1. On huggingface.co, create a new Space and choose Docker as the SDK.
2. Upload all files in this folder. The `Dockerfile` is used automatically.
3. In the Space's Settings, add a secret named `TWELVEDATA_API_KEY`.

Free plans sleep when nobody visits, so the first load can take up to a minute.
Keep to one timeframe most of the time: each timeframe uses its own share of
the 800 free Twelve Data requests a day.

## Files

- `ict_engine.py` - all ICT logic and the signal model (settings in `EngineConfig`)
- `data_sources.py` - Exness MT5, Twelve Data and demo feeds
- `app.py` - web server
- `templates/index.html` - dashboard
- `backtest.py` - historical testing

## Limits to know

- The app does not know about news. Stand aside around CPI, NFP and FOMC.
- Signals update when a candle closes, so an M5 signal can appear up to
  5 minutes after the move that created it.
- ICT concepts are not proven to give an edge by themselves. Trust your own
  backtest and demo results, not the score.
