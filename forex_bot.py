#!/usr/bin/env python3
"""
Forex Overextension Bot — GitHub Actions edition
• Exécuté une seule fois par run (le cron GH Actions remplace la boucle infinie)
• L'état anti-doublon est persisté via le cache GitHub Actions entre les runs
"""

import asyncio
import io
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # backend non-interactif (headless CI)
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv   # no-op si .env absent (OK en CI)
from telegram import Bot

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],   # stdout → visible dans GH Actions logs
)
log = logging.getLogger(__name__)

# ─── Credentials (injectés via Secrets GH Actions ou .env local) ─────────────
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID", "")

# ─── Univers des paires ───────────────────────────────────────────────────────
# yf : ticker yfinance (source gratuite)
# tv : symbole URL encodé pour le lien TradingView
FOREX_PAIRS: dict[str, dict] = {

    # ── Or ─────────────────────────────────────────────────────────────────
    "XAU/USD":  {"yf": "GC=F",       "tv": "OANDA%3AXAUUSD"},

    # ── Majeurs ────────────────────────────────────────────────────────────
    "EUR/USD":  {"yf": "EURUSD=X",   "tv": "FX%3AEURUSD"},
    "AUD/USD":  {"yf": "AUDUSD=X",   "tv": "FX%3AAUDUSD"},
    "NZD/USD":  {"yf": "NZDUSD=X",   "tv": "FX%3ANZDUSD"},
    "USD/CAD":  {"yf": "USDCAD=X",   "tv": "FX%3AUSDCAD"},
    "USD/CHF":  {"yf": "USDCHF=X",   "tv": "FX%3AUSDCHF"},
    "USD/JPY":  {"yf": "USDJPY=X",   "tv": "FX%3AUSDJPY"},
    "GBP/USD":  {"yf": "GBPUSD=X",   "tv": "FX%3AGBPUSD"},

    # ── Croisées EUR ───────────────────────────────────────────────────────
    "EUR/GBP":  {"yf": "EURGBP=X",   "tv": "FX%3AEURGBP"},
    "EUR/AUD":  {"yf": "EURAUD=X",   "tv": "FX%3AEURAUD"},
    "EUR/CAD":  {"yf": "EURCAD=X",   "tv": "FX%3AEURCAD"},
    "EUR/JPY":  {"yf": "EURJPY=X",   "tv": "FX%3AEURJPY"},
    "EUR/CHF":  {"yf": "EURCHF=X",   "tv": "FX%3AEURCHF"},
    "EUR/NZD":  {"yf": "EURNZD=X",   "tv": "FX%3AEURNZD"},

    # ── Croisées GBP ───────────────────────────────────────────────────────
    "GBP/JPY":  {"yf": "GBPJPY=X",   "tv": "FX%3AGBPJPY"},
    "GBP/AUD":  {"yf": "GBPAUD=X",   "tv": "FX%3AGBPAUD"},
    "GBP/CAD":  {"yf": "GBPCAD=X",   "tv": "FX%3AGBPCAD"},
    "GBP/CHF":  {"yf": "GBPCHF=X",   "tv": "FX%3AGBPCHF"},
    "GBP/NZD":  {"yf": "GBPNZD=X",   "tv": "FX%3AGBPNZD"},

    # ── Croisées AUD ───────────────────────────────────────────────────────
    "AUD/CAD":  {"yf": "AUDCAD=X",   "tv": "FX%3AAUDCAD"},
    "AUD/JPY":  {"yf": "AUDJPY=X",   "tv": "FX%3AAUDJPY"},
    "AUD/CHF":  {"yf": "AUDCHF=X",   "tv": "FX%3AAUDCHF"},
    "AUD/NZD":  {"yf": "AUDNZD=X",   "tv": "FX%3AAUDNZD"},

    # ── Autres croisées ────────────────────────────────────────────────────
    "CAD/JPY":  {"yf": "CADJPY=X",   "tv": "FX%3ACADJPY"},
    "CAD/CHF":  {"yf": "CADCHF=X",   "tv": "FX%3ACADCHF"},
    "CHF/JPY":  {"yf": "CHFJPY=X",   "tv": "FX%3ACHFJPY"},
    "NZD/JPY":  {"yf": "NZDJPY=X",   "tv": "FX%3ANZDJPY"},
    "NZD/CHF":  {"yf": "NZDCHF=X",   "tv": "FX%3ANZDCHF"},
    "NZD/CAD":  {"yf": "NZDCAD=X",   "tv": "FX%3ANZDCAD"},
}

# ─── Paramètres de détection ──────────────────────────────────────────────────
RSI_PERIOD          = 14
ATR_PERIOD          = 14
EMA_FAST            = 20
EMA_SLOW            = 50

RSI_OVERBOUGHT      = 72     # RSI > seuil → excès haussier
RSI_OVERSOLD        = 28     # RSI < seuil → excès baissier

ATR_MULT_IMPULSE    = 2.0    # Impulsion doit dépasser N × ATR
ATR_MULT_EMA_DIST   = 1.5    # Distance EMA doit dépasser N × ATR

IMPULSE_WINDOW      = 6      # Bougies sur lesquelles mesurer l'impulsion
MIN_DIRECTIONAL     = 4      # Minimum de bougies dans la même direction
MAX_RETRACE_RATIO   = 0.35   # Retracement max toléré (35 % du mouvement)

COOLDOWN_HOURS      = 4      # Délai avant de re-alerter même paire/direction
CHART_CANDLES       = 72     # Bougies H1 affichées sur le graphique (~3 jours)

ALERT_STATE_FILE    = Path("alert_state.json")


# ─── Indicateurs techniques ───────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ─── Récupération des données ─────────────────────────────────────────────────

def fetch_h1_data(yf_ticker: str) -> pd.DataFrame | None:
    """Télécharge 15 jours de données H1 via yfinance (source gratuite)."""
    try:
        df = yf.download(
            yf_ticker,
            period="15d",
            interval="1h",
            progress=False,
            auto_adjust=True,
        )
        if df.empty or len(df) < 60:
            log.warning(f"Données insuffisantes pour {yf_ticker} ({len(df)} bougies)")
            return None

        # Aplatir le MultiIndex (yfinance ≥ 0.2.x retourne un MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df

    except Exception as exc:
        log.error(f"Erreur fetch {yf_ticker}: {exc}")
        return None


# ─── Détection de l'overextension ────────────────────────────────────────────

def detect_overextension(df: pd.DataFrame) -> dict | None:
    """
    Retourne un dict si overextension détectée, sinon None.

    5 critères cumulatifs :
      1. RSI extrême         → RSI > RSI_OVERBOUGHT  ou  < RSI_OVERSOLD
      2. Forte impulsion     → amplitude > ATR_MULT_IMPULSE × ATR
      3. Éloigné de l'EMA   → |prix − EMA20| > ATR_MULT_EMA_DIST × ATR
      4. Mouvement directionnel → ≥ MIN_DIRECTIONAL bougies dans le même sens
      5. Retracement minimal → ratio pullback/impulse < MAX_RETRACE_RATIO
    """
    df = df.copy()
    df["RSI"]      = compute_rsi(df["Close"], RSI_PERIOD)
    df["ATR"]      = compute_atr(df, ATR_PERIOD)
    df["EMA_fast"] = compute_ema(df["Close"], EMA_FAST)
    df["EMA_slow"] = compute_ema(df["Close"], EMA_SLOW)

    last   = df.iloc[-1]
    window = df.iloc[-(IMPULSE_WINDOW + 1):]   # +1 pour inclure la bougie de départ

    rsi      = last["RSI"]
    atr      = last["ATR"]
    price    = last["Close"]
    ema_fast = last["EMA_fast"]

    if pd.isna(rsi) or pd.isna(atr) or atr == 0:
        return None

    # 1 — Direction via RSI
    if rsi > RSI_OVERBOUGHT:
        direction = "bullish"
    elif rsi < RSI_OVERSOLD:
        direction = "bearish"
    else:
        return None

    # 2 — Amplitude de l'impulsion
    impulse_start = window.iloc[0]["Close"]
    impulse       = abs(price - impulse_start)
    if impulse < ATR_MULT_IMPULSE * atr:
        return None

    # 3 — Distance EMA rapide
    ema_dist = abs(price - ema_fast)
    if ema_dist < ATR_MULT_EMA_DIST * atr:
        return None

    # 4 — Bougies directionnelles
    candles       = window.iloc[1:]
    bullish_count = (candles["Close"] > candles["Open"]).sum()
    bearish_count = (candles["Close"] < candles["Open"]).sum()

    if direction == "bullish" and bullish_count < MIN_DIRECTIONAL:
        return None
    if direction == "bearish" and bearish_count < MIN_DIRECTIONAL:
        return None

    # 5 — Retracement minimal
    if direction == "bullish":
        pullback = candles["High"].max() - candles["Low"].min()
    else:
        pullback = candles["High"].max() - candles["Low"].min()

    retrace_ratio = (pullback / impulse) if impulse > 0 else 1.0
    if retrace_ratio > MAX_RETRACE_RATIO:
        return None

    return {
        "direction":     direction,
        "rsi":           round(float(rsi), 1),
        "atr":           round(float(atr), 6),
        "impulse":       round(float(impulse), 6),
        "impulse_atr":   round(float(impulse / atr), 2),
        "ema_dist_atr":  round(float(ema_dist / atr), 2),
        "price":         round(float(price), 5),
        "retrace_ratio": round(float(retrace_ratio), 2),
    }


# ─── Génération du graphique ──────────────────────────────────────────────────

def generate_chart(df: pd.DataFrame, pair: str, direction: str) -> bytes:
    """Chandelier japonais H1, style sombre TradingView, EMA20 + EMA50."""
    chart_df = df.tail(CHART_CANDLES).copy()
    ema20    = compute_ema(chart_df["Close"], EMA_FAST)
    ema50    = compute_ema(chart_df["Close"], EMA_SLOW)

    add_plots = [
        mpf.make_addplot(ema20, color="#2196F3", width=1.4, label=f"EMA {EMA_FAST}"),
        mpf.make_addplot(ema50, color="#FF9800", width=1.4, label=f"EMA {EMA_SLOW}"),
    ]

    mc = mpf.make_marketcolors(
        up="#26a69a", down="#ef5350",
        edge="inherit", wick="inherit", volume="inherit",
    )

    try:
        mpf.make_mpf_style(base_mpf_style="nightclouds")
        base_style = "nightclouds"
    except Exception:
        base_style = "default"

    style = mpf.make_mpf_style(
        marketcolors=mc,
        base_mpf_style=base_style,
        gridstyle=":",
        gridcolor="#2A2A3A",
        facecolor="#131722",
        figcolor="#131722",
        rc={
            "axes.labelcolor": "#D1D4DC",
            "xtick.color":     "#D1D4DC",
            "ytick.color":     "#D1D4DC",
            "font.size":       10,
        },
    )

    label_dir = "BULLISH 🔼" if direction == "bullish" else "BEARISH 🔽"
    title     = f"\n{pair}  ·  H1  ·  Overextension {label_dir}"

    buf = io.BytesIO()
    fig, axes = mpf.plot(
        chart_df,
        type="candle",
        style=style,
        addplot=add_plots,
        title=title,
        figsize=(14, 7),
        returnfig=True,
        tight_layout=True,
        warn_too_much_data=300,
        volume=True,
        volume_panel=1,
        panel_ratios=(4, 1),
    )
    axes[0].title.set_color("#FFFFFF")
    axes[0].title.set_fontsize(13)
    fig.patch.set_facecolor("#131722")

    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130, facecolor="#131722")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─── Gestion de l'état anti-doublon ──────────────────────────────────────────
# En GitHub Actions, alert_state.json est persisté via le cache entre les runs.

def load_alert_state() -> dict:
    if ALERT_STATE_FILE.exists():
        try:
            return json.loads(ALERT_STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_alert_state(state: dict) -> None:
    ALERT_STATE_FILE.write_text(json.dumps(state, indent=2))


def is_on_cooldown(state: dict, pair: str, direction: str) -> bool:
    key = f"{pair}_{direction}"
    if key not in state:
        return False
    last_alert = datetime.fromisoformat(state[key])
    return datetime.utcnow() - last_alert < timedelta(hours=COOLDOWN_HOURS)


def mark_alerted(state: dict, pair: str, direction: str) -> None:
    state[f"{pair}_{direction}"] = datetime.utcnow().isoformat()


# ─── Envoi Telegram ───────────────────────────────────────────────────────────

async def send_alert(
    bot: Bot,
    pair: str,
    result: dict,
    tv_symbol: str,
    chart_bytes: bytes,
) -> None:
    direction  = result["direction"]
    emoji_main = "🔥" if direction == "bullish" else "❄️"
    arrow      = "🔼" if direction == "bullish" else "🔽"
    tv_url     = f"https://fr.tradingview.com/chart/?symbol={tv_symbol}"

    caption = (
        f"*New overextension on {pair} {emoji_main}*\n\n"
        f"{arrow} *Direction :* {direction.capitalize()}\n"
        f"📊 *RSI :* `{result['rsi']}`\n"
        f"📏 *Impulsion :* `{result['impulse_atr']}× ATR`\n"
        f"📐 *Distance EMA{EMA_FAST} :* `{result['ema_dist_atr']}× ATR`\n"
        f"💰 *Prix :* `{result['price']}`\n\n"
        f"[📈 Voir sur TradingView]({tv_url})"
    )

    await bot.send_photo(
        chat_id=TELEGRAM_CHANNEL_ID,
        photo=chart_bytes,
        caption=caption,
        parse_mode="Markdown",
    )
    log.info(
        f"✅ Alerte envoyée : {pair} {direction} "
        f"| RSI={result['rsi']} | {result['impulse_atr']}×ATR"
    )


# ─── Scan unique (appelé une fois par run GitHub Actions) ────────────────────

async def scan_all(bot: Bot) -> None:
    log.info("=" * 60)
    log.info(f"Scan démarré — {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    log.info(f"Paires surveillées : {len(FOREX_PAIRS)}")
    log.info("=" * 60)

    state       = load_alert_state()
    alerts_sent = 0

    for pair, info in FOREX_PAIRS.items():
        try:
            df = fetch_h1_data(info["yf"])
            if df is None:
                continue

            result = detect_overextension(df)
            if result is None:
                log.info(f"  {pair:<12} → ok, aucune overextension")
                continue

            direction = result["direction"]
            log.info(
                f"  {pair:<12} → 🚨 OVEREXTENSION {direction.upper()} "
                f"| RSI={result['rsi']} | {result['impulse_atr']}×ATR "
                f"| EMA dist={result['ema_dist_atr']}×ATR"
            )

            if is_on_cooldown(state, pair, direction):
                log.info(f"  {pair:<12} → cooldown actif ({COOLDOWN_HOURS}h), pas d'alerte")
                continue

            chart_bytes = generate_chart(df, pair, direction)
            await send_alert(bot, pair, result, info["tv"], chart_bytes)
            mark_alerted(state, pair, direction)
            save_alert_state(state)
            alerts_sent += 1

            await asyncio.sleep(1.5)   # éviter le flood Telegram

        except Exception as exc:
            log.error(f"  {pair:<12} → erreur inattendue : {exc}", exc_info=True)

    log.info("-" * 60)
    log.info(f"Scan terminé — {alerts_sent} alerte(s) envoyée(s)")


# ─── Point d'entrée ───────────────────────────────────────────────────────────
# GitHub Actions déclenche ce script toutes les 15 min via le cron du workflow.
# En local, pour tourner en continu, utilise run_local.py.

async def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN manquant (Secrets GH Actions ou .env)")
    if not TELEGRAM_CHANNEL_ID:
        raise ValueError("TELEGRAM_CHANNEL_ID manquant (Secrets GH Actions ou .env)")

    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    me  = await bot.get_me()
    log.info(f"Bot : @{me.username}  |  Channel : {TELEGRAM_CHANNEL_ID}")

    await scan_all(bot)


if __name__ == "__main__":
    asyncio.run(main())
