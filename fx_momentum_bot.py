#!/usr/bin/env python3
"""
FXMomentumBot — Alertes momentum Forex/Crypto/Indices
• Détecte les tendances qui s'accélèrent (≠ overextension qui cherche les retournements)
• Logique : MACD crossover (obligatoire) + ≥ 1 signal parmi : alignement EMA, RSI momentum, ROC
• Cooldown 4h par paire/direction pour éviter les doublons
"""

import asyncio
import io
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

TZ_PARIS = ZoneInfo("Europe/Paris")

def _now_paris() -> datetime:
    return datetime.now(TZ_PARIS)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
class _ParisFormatter(logging.Formatter):
    def converter(self, timestamp):
        return datetime.fromtimestamp(timestamp, TZ_PARIS).timetuple()

_handler = logging.StreamHandler()
_handler.setFormatter(_ParisFormatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_handler])
log = logging.getLogger(__name__)

# ─── Credentials ──────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
MOMENTUM_CHANNEL_ID  = os.getenv("MOMENTUM_CHANNEL_ID", os.getenv("TELEGRAM_CHANNEL_ID", ""))

# ─── Univers des paires ───────────────────────────────────────────────────────
FOREX_PAIRS: dict[str, dict] = {

    # ── Crypto ─────────────────────────────────────────────────────────────
    "BTC/USD":  {"yf": "BTC-USD",     "tv": "BITSTAMP%3ABTCUSD"},

    # ── Indices US ─────────────────────────────────────────────────────────
    "US30":     {"yf": "YM=F",        "tv": "OANDA%3AUS30USD"},
    "NAS100":   {"yf": "NQ=F",        "tv": "OANDA%3ANAS100USD"},
    "SPX500":   {"yf": "ES=F",        "tv": "OANDA%3ASPX500USD"},

    # ── Matières premières ─────────────────────────────────────────────────
    "XAU/USD":  {"yf": "GC=F",        "tv": "OANDA%3AXAUUSD"},
    "WTI/USD":  {"yf": "CL=F",        "tv": "NYMEX%3ACL1%21"},

    # ── Majeurs ────────────────────────────────────────────────────────────
    "EUR/USD":  {"yf": "EURUSD=X",    "tv": "FX%3AEURUSD"},
    "AUD/USD":  {"yf": "AUDUSD=X",    "tv": "FX%3AAUDUSD"},
    "USD/CAD":  {"yf": "USDCAD=X",    "tv": "FX%3AUSDCAD"},
    "USD/CHF":  {"yf": "USDCHF=X",    "tv": "FX%3AUSDCHF"},
    "USD/JPY":  {"yf": "USDJPY=X",    "tv": "FX%3AUSDJPY"},
    "GBP/USD":  {"yf": "GBPUSD=X",    "tv": "FX%3AGBPUSD"},

    # ── Croisées EUR ───────────────────────────────────────────────────────
    "EUR/GBP":  {"yf": "EURGBP=X",    "tv": "FX%3AEURGBP"},
    "EUR/AUD":  {"yf": "EURAUD=X",    "tv": "FX%3AEURAUD"},
    "EUR/CAD":  {"yf": "EURCAD=X",    "tv": "FX%3AEURCAD"},
    "EUR/JPY":  {"yf": "EURJPY=X",    "tv": "FX%3AEURJPY"},
    "EUR/CHF":  {"yf": "EURCHF=X",    "tv": "FX%3AEURCHF"},

    # ── Croisées GBP ───────────────────────────────────────────────────────
    "GBP/JPY":  {"yf": "GBPJPY=X",    "tv": "FX%3AGBPJPY"},
    "GBP/AUD":  {"yf": "GBPAUD=X",    "tv": "FX%3AGBPAUD"},
    "GBP/CAD":  {"yf": "GBPCAD=X",    "tv": "FX%3AGBPCAD"},
    "GBP/CHF":  {"yf": "GBPCHF=X",    "tv": "FX%3AGBPCHF"},

    # ── Croisées AUD ───────────────────────────────────────────────────────
    "AUD/CAD":  {"yf": "AUDCAD=X",    "tv": "FX%3AAUDCAD"},
    "AUD/JPY":  {"yf": "AUDJPY=X",    "tv": "FX%3AAUDJPY"},
    "AUD/CHF":  {"yf": "AUDCHF=X",    "tv": "FX%3AAUDCHF"},

    # ── Croisées NZD ───────────────────────────────────────────────────────
    "NZD/USD":  {"yf": "NZDUSD=X",   "tv": "FX%3ANZDUSD"},
    "EUR/NZD":  {"yf": "EURNZD=X",   "tv": "FX%3AEURNZD"},
    "GBP/NZD":  {"yf": "GBPNZD=X",   "tv": "FX%3AGBPNZD"},
    "AUD/NZD":  {"yf": "AUDNZD=X",   "tv": "FX%3AAUDNZD"},
    "NZD/JPY":  {"yf": "NZDJPY=X",   "tv": "FX%3ANZDJPY"},
    "NZD/CHF":  {"yf": "NZDCHF=X",   "tv": "FX%3ANZDCHF"},
    "NZD/CAD":  {"yf": "NZDCAD=X",   "tv": "FX%3ANZDCAD"},

    # ── Autres croisées ────────────────────────────────────────────────────
    "CAD/JPY":  {"yf": "CADJPY=X",    "tv": "FX%3ACADJPY"},
    "CAD/CHF":  {"yf": "CADCHF=X",    "tv": "FX%3ACADCHF"},
    "CHF/JPY":  {"yf": "CHFJPY=X",    "tv": "FX%3ACHFJPY"},
}

# ─── Paramètres de détection ──────────────────────────────────────────────────
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9
MACD_CROSS_WINDOW = 3    # crossover détecté si survenu dans les N dernières bougies

EMA_FAST         = 20
EMA_SLOW         = 50

RSI_PERIOD       = 14
RSI_MOM_BULL_MIN = 52    # RSI en zone momentum haussier
RSI_MOM_BULL_MAX = 70
RSI_MOM_BEAR_MIN = 30
RSI_MOM_BEAR_MAX = 48

ROC_PERIOD       = 10    # Rate of Change sur N bougies
ROC_THRESHOLD    = 0.3   # % minimum pour qualifier (0.3 % = 30 pips sur EUR/USD)

COOLDOWN_HOURS   = 4
CHART_RIGHT_MARGIN = 12

ALERT_STATE_FILE = Path("momentum_state.json")


# ─── Indicateurs ──────────────────────────────────────────────────────────────

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast   = compute_ema(series, MACD_FAST)
    slow   = compute_ema(series, MACD_SLOW)
    macd   = fast - slow
    signal = compute_ema(macd, MACD_SIGNAL)
    hist   = macd - signal
    return macd, signal, hist


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_roc(series: pd.Series, period: int) -> pd.Series:
    return ((series - series.shift(period)) / series.shift(period)) * 100


# ─── Session HTTP ─────────────────────────────────────────────────────────────
_YF_SESSION = requests.Session()
_YF_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
})


# ─── Récupération des données ─────────────────────────────────────────────────

def fetch_h1_data(yf_ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            yf_ticker,
            period="15d",
            interval="1h",
            progress=False,
            auto_adjust=True,
            session=_YF_SESSION,
        )
        if df.empty or len(df) < 60:
            log.warning(f"Données insuffisantes pour {yf_ticker} ({len(df)} bougies)")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as exc:
        log.error(f"Erreur fetch {yf_ticker}: {exc}")
        return None


# ─── Détection du momentum ────────────────────────────────────────────────────

def _strength_stars(n: int, total: int = 3) -> str:
    color = "🔴" if n == 1 else ("🟢" if n >= total else "🟠")
    return f"{color} {'★' * n}{'☆' * (total - n)}"


def detect_momentum(df: pd.DataFrame) -> dict:
    """
    Momentum confirmé si :
      ① MACD crossover dans les N dernières bougies  (condition obligatoire)
      + au moins 1 parmi :
      ② Alignement EMA20/EMA50 (prix du bon côté)
      ③ RSI en zone momentum (52–70 bull | 30–48 bear)
      ④ ROC > seuil dans la bonne direction
    """
    df = df.copy()
    df["EMA_fast"] = compute_ema(df["Close"], EMA_FAST)
    df["EMA_slow"] = compute_ema(df["Close"], EMA_SLOW)
    df["RSI"]      = compute_rsi(df["Close"], RSI_PERIOD)
    df["ROC"]      = compute_roc(df["Close"], ROC_PERIOD)
    df["MACD"], df["Signal"], df["Hist"] = compute_macd(df["Close"])

    base: dict = {
        "detected":      False,
        "reject_reason": "",
        "price":         round(float(df["Close"].iloc[-1]), 5),
        "rsi":           round(float(df["RSI"].iloc[-1]), 1) if not pd.isna(df["RSI"].iloc[-1]) else 0,
        "roc":           round(float(df["ROC"].iloc[-1]), 3) if not pd.isna(df["ROC"].iloc[-1]) else 0,
        "macd":          round(float(df["MACD"].iloc[-1]), 6) if not pd.isna(df["MACD"].iloc[-1]) else 0,
    }

    last = df.iloc[-1]
    if pd.isna(last["MACD"]) or pd.isna(last["RSI"]):
        base["reject_reason"] = "indicateurs invalides"
        return base

    # ── ① MACD crossover récent ───────────────────────────────────────────
    window = df.iloc[-(MACD_CROSS_WINDOW + 1):]
    hist_vals = window["Hist"].values

    bull_cross = any(hist_vals[i - 1] < 0 and hist_vals[i] >= 0 for i in range(1, len(hist_vals)))
    bear_cross = any(hist_vals[i - 1] > 0 and hist_vals[i] <= 0 for i in range(1, len(hist_vals)))

    if not bull_cross and not bear_cross:
        base["reject_reason"] = "pas de MACD crossover récent"
        return base

    direction = "bullish" if bull_cross else "bearish"

    # ── ② Alignement EMA ──────────────────────────────────────────────────
    price    = float(last["Close"])
    ema_fast = float(last["EMA_fast"])
    ema_slow = float(last["EMA_slow"])

    ema_aligned_bull = ema_fast > ema_slow and price > ema_fast
    ema_aligned_bear = ema_fast < ema_slow and price < ema_fast

    # ── ③ RSI momentum ────────────────────────────────────────────────────
    rsi = float(last["RSI"])
    rsi_mom_bull = RSI_MOM_BULL_MIN <= rsi <= RSI_MOM_BULL_MAX
    rsi_mom_bear = RSI_MOM_BEAR_MIN <= rsi <= RSI_MOM_BEAR_MAX

    # ── ④ ROC ─────────────────────────────────────────────────────────────
    roc = float(last["ROC"]) if not pd.isna(last["ROC"]) else 0.0
    roc_bull = roc >  ROC_THRESHOLD
    roc_bear = roc < -ROC_THRESHOLD

    # ── Construire les signaux confirmant la direction ────────────────────
    signals = []
    if direction == "bullish":
        signals.append(f"MACD crossover haussier")
        if ema_aligned_bull: signals.append(f"EMA{EMA_FAST} > EMA{EMA_SLOW}, prix > EMA{EMA_FAST}")
        if rsi_mom_bull:     signals.append(f"RSI {rsi:.1f} (zone momentum)")
        if roc_bull:         signals.append(f"ROC +{roc:.2f}%")
    else:
        signals.append(f"MACD crossover baissier")
        if ema_aligned_bear: signals.append(f"EMA{EMA_FAST} < EMA{EMA_SLOW}, prix < EMA{EMA_FAST}")
        if rsi_mom_bear:     signals.append(f"RSI {rsi:.1f} (zone momentum)")
        if roc_bear:         signals.append(f"ROC {roc:.2f}%")

    # Au moins 1 signal de confirmation en plus du MACD
    if len(signals) < 2:
        base["reject_reason"] = "MACD seul, pas de confirmation"
        return base

    base.update({
        "detected":     True,
        "direction":    direction,
        "signals":      signals,
        "strength":     len(signals),
        "strength_bar": _strength_stars(len(signals) - 1),  # -1 car MACD est obligatoire
    })
    return base


# ─── Génération du graphique ──────────────────────────────────────────────────

def generate_chart(df: pd.DataFrame, pair: str, direction: str, macd_col: pd.Series, signal_col: pd.Series) -> bytes:
    from datetime import timezone

    now      = datetime.now(timezone.utc)
    start_dt = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=2)
    if df.index.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=None)

    ema20_full = compute_ema(df["Close"], EMA_FAST)
    ema50_full = compute_ema(df["Close"], EMA_SLOW)

    chart_df   = df[df.index >= start_dt].copy()
    if chart_df.empty:
        chart_df = df.tail(48).copy()

    ema20  = ema20_full[chart_df.index]
    ema50  = ema50_full[chart_df.index]
    macd_c = macd_col[chart_df.index]
    sig_c  = signal_col[chart_df.index]

    add_plots = [
        mpf.make_addplot(ema20,   color="#FFFFFF",  width=1.2, label=f"EMA {EMA_FAST}"),
        mpf.make_addplot(ema50,   color="#F0A500",  width=1.2, label=f"EMA {EMA_SLOW}"),
        mpf.make_addplot(macd_c,  color="#2196F3",  width=1.0, panel=1, label="MACD"),
        mpf.make_addplot(sig_c,   color="#FF6B6B",  width=1.0, panel=1, label="Signal"),
    ]

    mc = mpf.make_marketcolors(up="#26a69a", down="#ef5350", edge="inherit", wick="inherit")
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
        title=f"\n{pair}  ·  H1  ·  Momentum {label_dir}",
        figsize=(14, 8),
        returnfig=True,
        tight_layout=True,
        warn_too_much_data=300,
        volume=False,
        panel_ratios=(3, 1),
    )

    ax = axes[0]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    for i, ts in enumerate(chart_df.index):
        if ts.hour == 0 and ts.minute == 0:
            ax.axvline(x=i, color="#4A4E6A", linewidth=0.9, linestyle="--", alpha=0.85, zorder=1)
            ax.text(i + 0.3, ymax, ts.strftime("%d %b"), color="#6B7099", fontsize=7.5, va="top")

    ax.set_xlim(xmin, xmax + CHART_RIGHT_MARGIN)
    ax.axvline(x=xmax - 0.5, color="#778899", linewidth=1.1, linestyle=":", alpha=0.75, zorder=2)
    ax.text(xmax + 0.4, ymax, "  →  12h", color="#778899", fontsize=8, va="top")

    axes[0].title.set_color("#FFFFFF")
    axes[0].title.set_fontsize(13)
    fig.patch.set_facecolor("#131722")
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130, facecolor="#131722")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─── Gestion de l'état anti-doublon ──────────────────────────────────────────

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
    return f"{pair}|{direction}|{','.join(sorted(signals))}"


def is_on_cooldown(state: dict, pair: str, direction: str, signals: list) -> bool:
    key = _alert_key(pair, direction, signals)
    if key not in state:
        return False
    return datetime.utcnow() - datetime.fromisoformat(state[key]) < timedelta(hours=COOLDOWN_HOURS)


def mark_alerted(state: dict, pair: str, direction: str, signals: list) -> None:
    state[_alert_key(pair, direction, signals)] = datetime.utcnow().isoformat()


# ─── Envoi Telegram ───────────────────────────────────────────────────────────

async def send_alert(bot: Bot, pair: str, result: dict, tv_symbol: str, chart_bytes: bytes) -> None:
    direction  = result["direction"]
    emoji_main = "🚀" if direction == "bullish" else "📉"
    arrow      = "🔼" if direction == "bullish" else "🔽"
    tv_url     = f"https://fr.tradingview.com/chart/?symbol={tv_symbol}"

    signals_text = "\n".join(f"✅ `{s}`" for s in result["signals"])
    now_str      = _now_paris().strftime("%d/%m/%Y %H:%M")

    caption = (
        f"*Momentum détecté sur {pair} {emoji_main}*\n\n"
        f"🕐 `{now_str}`\n"
        f"{arrow} *Direction :* {direction.capitalize()}\n"
        f"💰 *Prix :* `{result['price']}`\n\n"
        f"📡 *Signaux :*\n{signals_text}\n\n"
        f"⚡ *Force :* {result['strength_bar']}"
    )

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("📈 Ouvrir dans TradingView", url=tv_url),
    ]])

    await bot.send_photo(
        chat_id=MOMENTUM_CHANNEL_ID,
        photo=chart_bytes,
        caption=caption,
        parse_mode="Markdown",
        reply_markup=keyboard,
    )
    log.info(f"✅ Alerte momentum envoyée : {pair} {direction} | RSI={result['rsi']} | ROC={result['roc']}%")


# ─── Scan ─────────────────────────────────────────────────────────────────────

async def scan_all(bot: Bot) -> None:
    log.info("=" * 60)
    log.info(f"[FXMomentumBot] Scan — {_now_paris().strftime('%Y-%m-%d %H:%M:%S')} (Paris)")
    log.info(f"Paires surveillées : {len(FOREX_PAIRS)}")
    log.info("─" * 60)
    log.info("CONDITIONS DE DÉCLENCHEMENT :")
    log.info(f"  ① MACD crossover dans les {MACD_CROSS_WINDOW} dernières bougies  [obligatoire]")
    log.info(f"  ② EMA{EMA_FAST} / EMA{EMA_SLOW} alignées dans la direction")
    log.info(f"  ③ RSI en zone momentum ({RSI_MOM_BULL_MIN}–{RSI_MOM_BULL_MAX} bull | {RSI_MOM_BEAR_MIN}–{RSI_MOM_BEAR_MAX} bear)")
    log.info(f"  ④ ROC({ROC_PERIOD}) > {ROC_THRESHOLD}% dans la direction")
    log.info(f"  → Minimum : ① + 1 confirmation parmi ②③④")
    log.info(f"  [COOLDOWN] {COOLDOWN_HOURS}h par paire/direction")
    log.info("=" * 60)

    state      = load_alert_state()
    total_sent = 0

    for pair, info in FOREX_PAIRS.items():
        try:
            df = fetch_h1_data(info["yf"])
            if df is None:
                log.info(f"  {pair:<12} | ⚠️  données indisponibles")
                continue

            result = detect_momentum(df)

            # Recalculer MACD pour le graphique (sur le df complet)
            macd_s, signal_s, _ = compute_macd(df["Close"])

            ok  = "✅"
            nok = "❌"
            log.info(
                f"  {pair:<12} | "
                f"Prix={result['price']}  RSI={result['rsi']}  ROC={result['roc']:+.3f}%  "
                f"MACD={result['macd']:+.6f}"
                + (f"  → {result['reject_reason']}" if not result["detected"] else "")
            )

            if result["detected"]:
                direction = result["direction"]
                log.info(
                    f"  {pair:<12} | 🚀 MOMENTUM {direction.upper()} "
                    f"— {result['strength_bar']} "
                    f"— signaux : {', '.join(result['signals'])}"
                )
                if not is_on_cooldown(state, pair, direction, result["signals"]):
                    chart_bytes = generate_chart(df, pair, direction, macd_s, signal_s)
                    await send_alert(bot, pair, result, info["tv"], chart_bytes)
                    mark_alerted(state, pair, direction, result["signals"])
                    save_alert_state(state)
                    total_sent += 1
                    await asyncio.sleep(1.5)
                else:
                    log.info(f"  {pair:<12} | 🔒 cooldown actif — rien envoyé")

        except Exception as exc:
            log.error(f"  {pair:<12} | 💥 erreur inattendue : {exc}", exc_info=True)

    # Séparateur épinglé
    if total_sent > 0:
        try:
            chat = await bot.get_chat(chat_id=MOMENTUM_CHANNEL_ID)
            if chat.pinned_message:
                await bot.unpin_chat_message(chat_id=MOMENTUM_CHANNEL_ID, message_id=chat.pinned_message.message_id)
        except Exception as e:
            log.warning(f"Impossible de désépingler : {e}")

        msg = await bot.send_message(chat_id=MOMENTUM_CHANNEL_ID, text="⚡" * 15)
        try:
            await bot.pin_chat_message(chat_id=MOMENTUM_CHANNEL_ID, message_id=msg.message_id, disable_notification=True)
        except Exception as e:
            log.warning(f"Impossible d'épingler : {e}")

    log.info("-" * 60)
    log.info(f"Scan terminé — {total_sent} alerte(s) envoyée(s)")


# ─── Point d'entrée ───────────────────────────────────────────────────────────

async def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN manquant")
    if not MOMENTUM_CHANNEL_ID:
        raise ValueError("MOMENTUM_CHANNEL_ID manquant (ou TELEGRAM_CHANNEL_ID)")

    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    me  = await bot.get_me()
    log.info(f"FXMomentumBot connecté : @{me.username}")
    log.info(f"Channel : {MOMENTUM_CHANNEL_ID}")
    await scan_all(bot)


if __name__ == "__main__":
    asyncio.run(main())
