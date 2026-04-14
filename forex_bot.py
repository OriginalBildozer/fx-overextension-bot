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

RSI_OVERBOUGHT      = 67     # RSI > seuil → excès haussier
RSI_OVERSOLD        = 33     # RSI < seuil → excès baissier

ATR_MULT_IMPULSE    = 1.0    # Impulsion doit dépasser N × ATR
ATR_MULT_EMA_DIST   = 1.0    # Distance EMA doit dépasser N × ATR

IMPULSE_WINDOW      = 6      # Bougies sur lesquelles mesurer l'impulsion
MAX_RETRACE_RATIO   = 0.20   # Retracement max toléré — condition ET obligatoire
# Logique : (RSI OU Impulsion OU EMA-dist)  ET  (retracement ≤ 20 %)

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

def _strength_bar(n: int, total: int = 3) -> str:
    """Barre de progression unicode. Ex : 2/3  [██░]"""
    return f"{n}/{total}  [{'█' * n}{'░' * (total - n)}]"


def detect_overextension(df: pd.DataFrame) -> dict | None:
    """
    Retourne un dict si overextension détectée, sinon None.

    Logique : (RSI OU Impulsion OU EMA-dist)  ET  retracement ≤ 20 %
      • RSI extrême    → RSI > 67  ou  < 33
      • Impulsion      → move net des 6 dernières bougies > 1× ATR
      • Distance EMA20 → |prix − EMA20| > 1× ATR
      • [ET] Retracement depuis l'extrême ≤ 20 % de l'impulsion (filtre obligatoire)

    Direction = côté ayant le plus de signaux OR ; égalité → priorité au RSI.
    Force du signal = nombre de critères OR déclenchés (1, 2 ou 3).
    """
    df = df.copy()
    df["RSI"]      = compute_rsi(df["Close"], RSI_PERIOD)
    df["ATR"]      = compute_atr(df, ATR_PERIOD)
    df["EMA_fast"] = compute_ema(df["Close"], EMA_FAST)
    df["EMA_slow"] = compute_ema(df["Close"], EMA_SLOW)

    last   = df.iloc[-1]
    window = df.iloc[-(IMPULSE_WINDOW + 1):]

    rsi      = last["RSI"]
    atr      = last["ATR"]
    price    = last["Close"]
    ema_fast = last["EMA_fast"]

    if pd.isna(rsi) or pd.isna(atr) or atr == 0:
        return None

    impulse_start   = window.iloc[0]["Close"]
    signed_impulse  = float(price - impulse_start)   # + = haussier
    ema_dist_signed = float(price - ema_fast)         # + = au-dessus EMA

    # ── Critères OR ───────────────────────────────────────────────────────
    bullish_signals, bearish_signals = [], []

    if rsi > RSI_OVERBOUGHT:
        bullish_signals.append(f"RSI {rsi:.1f} > {RSI_OVERBOUGHT}")
    elif rsi < RSI_OVERSOLD:
        bearish_signals.append(f"RSI {rsi:.1f} < {RSI_OVERSOLD}")

    if signed_impulse > ATR_MULT_IMPULSE * atr:
        bullish_signals.append(f"Impulsion +{signed_impulse/atr:.1f}×ATR")
    elif signed_impulse < -ATR_MULT_IMPULSE * atr:
        bearish_signals.append(f"Impulsion {signed_impulse/atr:.1f}×ATR")

    if ema_dist_signed > ATR_MULT_EMA_DIST * atr:
        bullish_signals.append(f"EMA dist +{ema_dist_signed/atr:.1f}×ATR")
    elif ema_dist_signed < -ATR_MULT_EMA_DIST * atr:
        bearish_signals.append(f"EMA dist {ema_dist_signed/atr:.1f}×ATR")

    if not bullish_signals and not bearish_signals:
        return None

    # Direction = côté le plus représenté ; égalité → RSI prioritaire
    if len(bullish_signals) >= len(bearish_signals):
        direction = "bullish"
        signals   = bullish_signals
    else:
        direction = "bearish"
        signals   = bearish_signals

    # ── Condition ET — retracement depuis l'extrême ≤ 20 % ───────────────
    impulse_abs = abs(signed_impulse)
    if impulse_abs > 0:
        if direction == "bullish":
            peak    = float(window["High"].max())
            retrace = (peak - float(price)) / impulse_abs
        else:
            trough  = float(window["Low"].min())
            retrace = (float(price) - trough) / impulse_abs
    else:
        retrace = 0.0   # pas d'impulsion mesurable → filtre inactif

    if retrace > MAX_RETRACE_RATIO:
        return None     # le mouvement a déjà trop corrigé

    strength = len(signals)   # 1, 2 ou 3

    return {
        "direction":    direction,
        "signals":      signals,
        "strength":     strength,
        "strength_bar": _strength_bar(strength),
        "rsi":          round(float(rsi), 1),
        "impulse_atr":  round(float(signed_impulse / atr), 2),
        "ema_dist_atr": round(float(ema_dist_signed / atr), 2),
        "retrace_pct":  round(retrace * 100, 1),
        "price":        round(float(price), 5),
        "atr":          round(float(atr), 6),
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


def _alert_key(pair: str, direction: str, signals: list) -> str:
    """Clé unique = paire + direction + ensemble exact des signaux.
    Si les signaux changent (même paire, même direction), la clé change
    → le cooldown ne s'applique pas → nouvelle alerte autorisée."""
    return f"{pair}|{direction}|{','.join(sorted(signals))}"


def is_on_cooldown(state: dict, pair: str, direction: str, signals: list) -> bool:
    key = _alert_key(pair, direction, signals)
    if key not in state:
        return False
    last_alert = datetime.fromisoformat(state[key])
    return datetime.utcnow() - last_alert < timedelta(hours=COOLDOWN_HOURS)


def mark_alerted(state: dict, pair: str, direction: str, signals: list) -> None:
    state[_alert_key(pair, direction, signals)] = datetime.utcnow().isoformat()


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

    signals_text = "\n".join(f"  ✅ `{s}`" for s in result["signals"])

    caption = (
        f"*New overextension on {pair} {emoji_main}*\n\n"
        f"{arrow} *Direction :* {direction.capitalize()}\n"
        f"💰 *Prix :* `{result['price']}`\n\n"
        f"*Signaux déclenchés :*\n{signals_text}\n\n"
        f"⚡ *Force du signal :* `{result['strength_bar']}`\n"
        f"↩️ *Retracement :* `{result['retrace_pct']} %`\n\n"
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

            if is_on_cooldown(state, pair, direction, result["signals"]):
                log.info(f"  {pair:<12} → cooldown actif (signaux identiques), pas d'alerte")
                continue

            chart_bytes = generate_chart(df, pair, direction)
            await send_alert(bot, pair, result, info["tv"], chart_bytes)
            mark_alerted(state, pair, direction, result["signals"])
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
