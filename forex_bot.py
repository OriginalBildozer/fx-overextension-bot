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
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

TZ_PARIS = ZoneInfo("Europe/Paris")

def _now_paris() -> datetime:
    """Heure courante en timezone Europe/Paris."""
    return datetime.now(TZ_PARIS)

import matplotlib
matplotlib.use("Agg")   # backend non-interactif (headless CI)
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv   # no-op si .env absent (OK en CI)
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

load_dotenv()

# ─── Logging (heure Paris) ────────────────────────────────────────────────────
class _ParisFormatter(logging.Formatter):
    def converter(self, timestamp):
        return datetime.fromtimestamp(timestamp, TZ_PARIS).timetuple()

_handler = logging.StreamHandler()
_handler.setFormatter(_ParisFormatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_handler])
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

ATR_MULT_IMPULSE    = 2.0    # Impulsion doit dépasser N × ATR
ATR_MULT_EMA_DIST   = 2.0    # Distance EMA doit dépasser N × ATR

IMPULSE_WINDOW      = 3      # Bougies sur lesquelles mesurer l'impulsion
MAX_RETRACE_RATIO   = 0.20   # Retracement max toléré — condition ET obligatoire
CANDLE_RANGE_LOOKBACK = 3    # Nombre de bougies précédentes pour la moyenne range
# Logique : (RSI OU Impulsion OU EMA-dist OU EMA+Range)  ET  (retracement ≤ 20 %)

COOLDOWN_HOURS      = 4      # Délai avant de re-alerter même paire/direction
CHART_RIGHT_MARGIN  = 12     # Espace vide à droite (12h à venir)

ALERT_STATE_FILE    = Path("alert_state.json")

# ─── Paramètres patterns (Pin Bar / Engulfing) ───────────────────────────────
PIN_BAR_BODY_MAX_PCT = 0.35   # Corps ≤ 35% du range total
PIN_BAR_WICK_MIN_PCT = 0.60   # Mèche dominante ≥ 60% du range total

PATTERN_TIMEFRAMES = {
    "M15": {"interval": "15m", "period": "5d"},
    "M30": {"interval": "30m", "period": "7d"},
    "H1":  {"interval": "1h",  "period": "15d"},
}



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

def _strength_stars(n: int, total: int = 4) -> str:
    """Étoiles colorées selon le nombre de conditions déclenchées.
    1 → rouge  |  2-3 → orange  |  4 → vert
    Ex : 🟠 ★★★☆
    """
    color = "🔴" if n == 1 else ("🟢" if n == total else "🟠")
    return f"{color} {'★' * n}{'☆' * (total - n)}"


def detect_overextension(df: pd.DataFrame) -> dict:
    """
    Retourne TOUJOURS un dict. Vérifier result["detected"] pour savoir
    si une overextension a été trouvée.

    Logique : (RSI OU Impulsion OU EMA-dist OU EMA+Range)  ET  retracement ≤ 20 %
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

    # Valeurs de base toujours retournées pour les logs
    base: dict = {
        "detected":        False,
        "reject_reason":   "",
        "price":           round(float(price), 5),
        "atr":             round(float(atr), 6) if not pd.isna(atr) else 0,
        "rsi":             round(float(rsi), 1) if not pd.isna(rsi) else 0,
        "impulse_atr":     0.0,
        "ema_dist_atr":    0.0,
        "candle_range_ratio": 0.0,   # range actuel / moyenne des N précédentes
        "retrace_pct":     0.0,
    }

    if pd.isna(rsi) or pd.isna(atr) or atr == 0:
        base["reject_reason"] = "indicateurs invalides"
        return base

    impulse_start   = window.iloc[0]["Close"]
    signed_impulse  = float(price - impulse_start)
    ema_dist_signed = float(price - ema_fast)

    base["impulse_atr"]  = round(float(signed_impulse / atr), 2)
    base["ema_dist_atr"] = round(float(ema_dist_signed / atr), 2)

    # ── Critères OR ───────────────────────────────────────────────────────
    bullish_signals, bearish_signals = [], []

    # 1 — RSI
    if rsi > RSI_OVERBOUGHT:
        bullish_signals.append(f"RSI {rsi:.1f} > {RSI_OVERBOUGHT}")
    elif rsi < RSI_OVERSOLD:
        bearish_signals.append(f"RSI {rsi:.1f} < {RSI_OVERSOLD}")

    # 2 — Impulsion
    if signed_impulse > ATR_MULT_IMPULSE * atr:
        bullish_signals.append(f"Impulsion +{signed_impulse/atr:.1f}×ATR")
    elif signed_impulse < -ATR_MULT_IMPULSE * atr:
        bearish_signals.append(f"Impulsion {signed_impulse/atr:.1f}×ATR")

    # 3 — Distance EMA20
    if ema_dist_signed > ATR_MULT_EMA_DIST * atr:
        bullish_signals.append(f"EMA dist +{ema_dist_signed/atr:.1f}×ATR")
    elif ema_dist_signed < -ATR_MULT_EMA_DIST * atr:
        bearish_signals.append(f"EMA dist {ema_dist_signed/atr:.1f}×ATR")

    # Calcul du range — utilisé par la condition 4 (EMA + Range)
    current_range  = float(last["High"] - last["Low"])
    prev_ranges    = [
        float(df.iloc[-i]["High"] - df.iloc[-i]["Low"])
        for i in range(2, 2 + CANDLE_RANGE_LOOKBACK)
    ]
    avg_prev_range = sum(prev_ranges) / len(prev_ranges) if prev_ranges else 0
    range_ratio    = (current_range / avg_prev_range) if avg_prev_range > 0 else 0.0
    base["candle_range_ratio"] = round(range_ratio, 2)

    # 4 — (EMA dist > 2×ATR) ET (bougie ≥ 1.5× moy range) simultanément
    ema_cond   = abs(ema_dist_signed) > ATR_MULT_EMA_DIST * atr
    range_cond = range_ratio >= 1.5
    if ema_cond and range_cond:
        if ema_dist_signed > 0:
            bullish_signals.append(
                f"EMA+Range ({ema_dist_signed/atr:+.2f}×ATR & ×{range_ratio:.2f} moy)"
            )
        else:
            bearish_signals.append(
                f"EMA+Range ({ema_dist_signed/atr:+.2f}×ATR & ×{range_ratio:.2f} moy)"
            )

    if not bullish_signals and not bearish_signals:
        base["reject_reason"] = "aucun signal OR"
        return base

    # Direction
    if len(bullish_signals) >= len(bearish_signals):
        direction = "bullish"
        signals   = bullish_signals
    else:
        direction = "bearish"
        signals   = bearish_signals

    # ── Condition ET — retracement ≤ 20 % ────────────────────────────────
    # Seuil minimum d'impulsion pour que le calcul ait du sens.
    # En-dessous, le HH/LL seul a déclenché sans mouvement net → filtre inactif.
    MIN_IMPULSE_FOR_RETRACE = 0.3   # × ATR

    impulse_abs = abs(signed_impulse)
    if impulse_abs >= MIN_IMPULSE_FOR_RETRACE * atr:
        if direction == "bullish":
            peak    = float(window["High"].max())
            retrace = (peak - float(price)) / impulse_abs
        else:
            trough  = float(window["Low"].min())
            retrace = (float(price) - trough) / impulse_abs
    else:
        retrace = 0.0   # impulsion trop faible → retracement non calculable

    base["retrace_pct"] = round(retrace * 100, 1)

    if retrace > MAX_RETRACE_RATIO:
        base["reject_reason"] = f"retracement {retrace*100:.1f}% > {MAX_RETRACE_RATIO*100:.0f}%"
        return base

    strength = len(signals)
    base.update({
        "detected":     True,
        "direction":    direction,
        "signals":      signals,
        "strength":     strength,
        "strength_bar": _strength_stars(strength),
    })
    return base


# ─── Génération du graphique ──────────────────────────────────────────────────

def generate_chart(df: pd.DataFrame, pair: str, direction: str) -> bytes:
    """
    Chandelier japonais H1, style sombre TradingView.
    - Fenêtre : minuit J-2 → maintenant + 12h vides à droite
    - EMA 20 blanche
    - Ligne pointillée à chaque minuit
    - Ligne pointillée séparant passé / futur
    """
    from datetime import timezone

    # ── Fenêtre : minuit d'il y a 2 jours ────────────────────────────────
    now      = datetime.now(timezone.utc)
    start_dt = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=2)

    # Harmoniser timezone avec l'index du DataFrame
    if df.index.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=None)

    # Calculer l'EMA sur tout le df (évite l'effet de chauffe en début de fenêtre)
    ema20_full = compute_ema(df["Close"], EMA_FAST)

    chart_df = df[df.index >= start_dt].copy()
    ema20    = ema20_full[chart_df.index]

    if chart_df.empty:
        chart_df = df.tail(48).copy()
        ema20    = ema20_full[chart_df.index]

    add_plots = [
        mpf.make_addplot(ema20, color="#FFFFFF", width=1.4, label=f"EMA {EMA_FAST}"),
    ]
    mc = mpf.make_marketcolors(
        up="#26a69a", down="#ef5350",
        edge="inherit", wick="inherit",
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
    buf = io.BytesIO()
    fig, axes = mpf.plot(
        chart_df,
        type="candle",
        style=style,
        addplot=add_plots,
        title=f"\n{pair}  ·  H1  ·  Overextension {label_dir}",
        figsize=(14, 7),
        returnfig=True,
        tight_layout=True,
        warn_too_much_data=300,
        volume=False,
    )

    ax = axes[0]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # ── Lignes verticales à chaque minuit ─────────────────────────────────
    for i, ts in enumerate(chart_df.index):
        if ts.hour == 0 and ts.minute == 0:
            ax.axvline(
                x=i,
                color="#4A4E6A",
                linewidth=0.9,
                linestyle="--",
                alpha=0.85,
                zorder=1,
            )
            ax.text(
                i + 0.3, ymax,
                ts.strftime("%d %b"),
                color="#6B7099",
                fontsize=7.5,
                va="top",
            )

    # ── Espace vide de 12h à droite + séparateur passé/futur ─────────────
    ax.set_xlim(xmin, xmax + CHART_RIGHT_MARGIN)
    ax.axvline(
        x=xmax - 0.5,
        color="#778899",
        linewidth=1.1,
        linestyle=":",
        alpha=0.75,
        zorder=2,
    )
    ax.text(
        xmax + 0.4, ymax,
        "  →  12h",
        color="#778899",
        fontsize=8,
        va="top",
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
    patterns: list | None = None,
) -> None:
    direction  = result["direction"]
    emoji_main = "🔥" if direction == "bullish" else "❄️"
    arrow      = "🔼" if direction == "bullish" else "🔽"
    tv_url_https = f"https://fr.tradingview.com/chart/?symbol={tv_symbol}"

    signals_text = "\n".join(f"  ✅ `{s}`" for s in result["signals"])

    now_str = _now_paris().strftime("%d/%m/%Y %H:%M")

    caption = (
        f"*New overextension on {pair} {emoji_main}*\n\n"
        f"🕐 `{now_str}`\n"
        f"{arrow} *Direction :* {direction.capitalize()}\n"
        f"💰 *Prix :* `{result['price']}`\n\n"
        f"*Signaux déclenchés :*\n{signals_text}\n\n"
        f"⚡ *Force du signal :* {result['strength_bar']}\n"
        f"↩️ *Retracement :* `{result['retrace_pct']} %`"
    )

    if patterns:
        pattern_lines = "\n".join(
            f"  📈🚀 *{p['pattern'].upper()} DETECTEE* `[{p['tf']}]` !!!"
            for p in patterns
        )
        caption += f"\n\n{pattern_lines}"

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("📈 Ouvrir dans TradingView", url=tv_url_https),
    ]])

    await bot.send_photo(
        chat_id=TELEGRAM_CHANNEL_ID,
        photo=chart_bytes,
        caption=caption,
        parse_mode="Markdown",
        reply_markup=keyboard,
    )
    log.info(
        f"✅ Alerte envoyée : {pair} {direction} "
        f"| RSI={result['rsi']} | {result['impulse_atr']}×ATR"
    )


# ─── Fetch multi-timeframe ───────────────────────────────────────────────────

def fetch_tf_data(yf_ticker: str, interval: str, period: str) -> pd.DataFrame | None:
    """Télécharge des données pour un timeframe quelconque."""
    try:
        df = yf.download(yf_ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 5:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as exc:
        log.error(f"Erreur fetch {yf_ticker} {interval}: {exc}")
        return None


# ─── Détection des patterns ───────────────────────────────────────────────────

def detect_pin_bar(df: pd.DataFrame) -> dict | None:
    """Détecte une pin bar sur la dernière bougie."""
    last = df.iloc[-1]
    o, h, l, c = float(last["Open"]), float(last["High"]), float(last["Low"]), float(last["Close"])
    total_range = h - l
    if total_range == 0:
        return None
    body        = abs(c - o)
    upper_wick  = h - max(o, c)
    lower_wick  = min(o, c) - l
    if body / total_range > PIN_BAR_BODY_MAX_PCT:
        return None
    if lower_wick / total_range >= PIN_BAR_WICK_MIN_PCT:
        return {"direction": "bullish", "pattern": "Pin Bar",
                "detail": f"mèche basse {lower_wick/total_range*100:.0f}% du range"}
    if upper_wick / total_range >= PIN_BAR_WICK_MIN_PCT:
        return {"direction": "bearish", "pattern": "Pin Bar",
                "detail": f"mèche haute {upper_wick/total_range*100:.0f}% du range"}
    return None


def detect_engulfing(df: pd.DataFrame) -> dict | None:
    """Détecte une bougie engulfing sur les 2 dernières bougies."""
    if len(df) < 2:
        return None
    prev, curr = df.iloc[-2], df.iloc[-1]
    po, pc = float(prev["Open"]), float(prev["Close"])
    co, cc = float(curr["Open"]), float(curr["Close"])
    prev_bull = pc > po
    curr_bull  = cc > co
    if prev_bull == curr_bull:
        return None
    # Le corps courant englobe entièrement le corps précédent
    if max(co, cc) > max(po, pc) and min(co, cc) < min(po, pc):
        direction = "bullish" if curr_bull else "bearish"
        ratio = abs(cc - co) / max(abs(pc - po), 1e-10)
        return {"direction": direction, "pattern": "Engulfing",
                "detail": f"corps ×{ratio:.2f} vs bougie préc."}
    return None


# ─── Détection patterns sur tous les timeframes ───────────────────────────────

def detect_patterns_all_tf(yf_ticker: str) -> list:
    """Vérifie Pin Bar et Engulfing sur M15, M30, H1.
    Retourne la liste des patterns trouvés (dédoublonnés par type)."""
    found = {}   # clé = pattern name → on garde 1 seul par type (le plus petit TF)
    for tf, tf_info in PATTERN_TIMEFRAMES.items():
        df = fetch_tf_data(yf_ticker, tf_info["interval"], tf_info["period"])
        if df is None:
            continue
        for detector in [detect_pin_bar, detect_engulfing]:
            pat = detector(df)
            if pat and pat["pattern"] not in found:
                found[pat["pattern"]] = {**pat, "tf": tf}
    return list(found.values())


# ─── Scan unique (appelé une fois par run GitHub Actions) ────────────────────

async def scan_all(bot: Bot) -> None:
    log.info("=" * 60)
    log.info(f"Scan démarré — {_now_paris().strftime('%Y-%m-%d %H:%M:%S')} (Paris)")
    log.info(f"Paires surveillées : {len(FOREX_PAIRS)}")
    log.info("─" * 60)
    log.info("CONDITIONS DE DÉCLENCHEMENT (logique OR + filtre ET) :")
    log.info(f"  ① RSI extrême     : RSI > {RSI_OVERBOUGHT} (haussier)  ou  RSI < {RSI_OVERSOLD} (baissier)  [période {RSI_PERIOD}]")
    log.info(f"  ② Impulsion forte : move net sur {IMPULSE_WINDOW} bougies  >  {ATR_MULT_IMPULSE}× ATR")
    log.info(f"  ③ Distance EMA    : |prix − EMA{EMA_FAST}|  >  {ATR_MULT_EMA_DIST}× ATR")
    log.info(f"  ④ EMA + Range     : (|prix − EMA{EMA_FAST}| > {ATR_MULT_EMA_DIST}×ATR)  ET  (range bougie ≥ 1.5× moy {CANDLE_RANGE_LOOKBACK} préc.)")
    log.info(f"  [ET] Retracement  : pullback depuis l'extrême  ≤  {int(MAX_RETRACE_RATIO*100)}% de l'impulsion")
    log.info(f"  [COOLDOWN]        : {COOLDOWN_HOURS}h par paire/direction/signaux identiques")
    log.info("=" * 60)

    state       = load_alert_state()
    alerts_sent = 0

    for pair, info in FOREX_PAIRS.items():
        try:
            df = fetch_h1_data(info["yf"])
            if df is None:
                log.info(f"  {pair:<12} | ⚠️  données indisponibles")
                continue

            result = detect_overextension(df)

            # ── Log détaillé des indicateurs ──────────────────────────────
            ok  = "✅"
            nok = "❌"
            rsi_ok     = ok if abs(result["rsi"] - 50) >= (50 - RSI_OVERSOLD)    else nok
            imp_ok     = ok if abs(result["impulse_atr"]) >= ATR_MULT_IMPULSE   else nok
            ema_ok     = ok if abs(result["ema_dist_atr"]) >= ATR_MULT_EMA_DIST else nok
            ema_rng_ok = ok if (abs(result["ema_dist_atr"]) >= ATR_MULT_EMA_DIST
                                and result["candle_range_ratio"] >= 1.5)        else nok

            log.info(
                f"  {pair:<12} | "
                f"Prix={result['price']}  ATR={result['atr']}  "
                f"RSI={result['rsi']}{rsi_ok}  "
                f"Imp={result['impulse_atr']:+.2f}×{imp_ok}  "
                f"EMA={result['ema_dist_atr']:+.2f}×{ema_ok}  "
                f"Range=×{result['candle_range_ratio']}  "
                f"EMA+Rng{ema_rng_ok}  "
                f"Retrace={result['retrace_pct']}%"
            )

            # ── Résultat de la détection ───────────────────────────────────
            if not result["detected"]:
                log.info(f"  {pair:<12} | ⛔ non déclenché — {result['reject_reason']}")
                continue

            direction = result["direction"]
            log.info(
                f"  {pair:<12} | 🚨 OVEREXTENSION {direction.upper()} "
                f"— {result['strength_bar']} "
                f"— signaux : {', '.join(result['signals'])}"
            )

            if is_on_cooldown(state, pair, direction, result["signals"]):
                log.info(f"  {pair:<12} | 🔒 cooldown actif (signaux identiques) — message non envoyé")
                continue

            patterns = detect_patterns_all_tf(info["yf"])
            if patterns:
                log.info(f"  {pair:<12} | 📊 patterns : " +
                         ", ".join(f"{p['pattern']} [{p['tf']}]" for p in patterns))

            chart_bytes = generate_chart(df, pair, direction)
            await send_alert(bot, pair, result, info["tv"], chart_bytes, patterns=patterns)
            mark_alerted(state, pair, direction, result["signals"])
            save_alert_state(state)
            alerts_sent += 1
            log.info(f"  {pair:<12} | ✅ message Telegram envoyé")

            await asyncio.sleep(1.5)

        except Exception as exc:
            log.error(f"  {pair:<12} | 💥 erreur inattendue : {exc}", exc_info=True)

    # Séparateur de fin de salve
    if alerts_sent > 0:
        await bot.send_message(
            chat_id=TELEGRAM_CHANNEL_ID,
            text="‼️" * 15,
        )

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
