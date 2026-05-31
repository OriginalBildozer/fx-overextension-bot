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

    # ── Autres croisées ────────────────────────────────────────────────────
    "CAD/JPY":  {"yf": "CADJPY=X",    "tv": "FX%3ACADJPY"},
    "CAD/CHF":  {"yf": "CADCHF=X",    "tv": "FX%3ACADCHF"},
    "CHF/JPY":  {"yf": "CHFJPY=X",    "tv": "FX%3ACHFJPY"},
}

# ─── Paramètres de détection ──────────────────────────────────────────────────
RSI_PERIOD          = 14
ATR_PERIOD          = 14
EMA_FAST            = 20
EMA_SLOW            = 50

RSI_OVERBOUGHT      = 67     # RSI > seuil → excès haussier
RSI_OVERSOLD        = 33     # RSI < seuil → excès baissier

ATR_MULT_IMPULSE    = 1.5    # Impulsion doit dépasser N × ATR
ATR_MULT_EMA_DIST   = 1.5    # Distance EMA doit dépasser N × ATR

IMPULSE_WINDOW      = 3      # Bougies sur lesquelles mesurer l'impulsion
CANDLE_RANGE_LOOKBACK = 3    # Nombre de bougies précédentes pour la moyenne range
# Logique : (RSI OU Impulsion OU EMA-dist OU EMA+Range)  ET  (retracement ≤ 20 %)

COOLDOWN_HOURS      = 4      # Délai avant de re-alerter même paire/direction
CHART_RIGHT_MARGIN  = 12     # Espace vide à droite (12h à venir)

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
    }

    if pd.isna(rsi) or pd.isna(atr) or atr == 0:
        base["reject_reason"] = "indicateurs invalides"
        return base

    impulse_start   = window.iloc[0]["Close"]
    signed_impulse  = float(price - impulse_start)
    ema_dist_signed = float(price - ema_fast)

    base["impulse_atr"]  = round(float(signed_impulse / atr), 2)
    base["ema_dist_atr"] = round(float(ema_dist_signed / atr), 2)

    # ── Évaluation de toutes les conditions ──────────────────────────────
    rsi_bull = rsi > RSI_OVERBOUGHT
    rsi_bear = rsi < RSI_OVERSOLD

    bullish_signals, bearish_signals = [], []

    # 2 — Impulsion
    imp_bull = signed_impulse > ATR_MULT_IMPULSE * atr
    imp_bear = signed_impulse < -ATR_MULT_IMPULSE * atr

    # 3 — Distance EMA20
    ema_bull = ema_dist_signed > ATR_MULT_EMA_DIST * atr
    ema_bear = ema_dist_signed < -ATR_MULT_EMA_DIST * atr

    # Calcul du range — condition 4
    current_range   = float(last["High"] - last["Low"])
    avg_before_curr = sum(
        float(df.iloc[-i]["High"] - df.iloc[-i]["Low"])
        for i in range(2, 2 + CANDLE_RANGE_LOOKBACK)
    ) / CANDLE_RANGE_LOOKBACK
    ratio_curr = (current_range / avg_before_curr) if avg_before_curr > 0 else 0.0

    prev_candle      = df.iloc[-2]
    prev_range       = float(prev_candle["High"] - prev_candle["Low"])
    avg_before_prev  = sum(
        float(df.iloc[-i]["High"] - df.iloc[-i]["Low"])
        for i in range(3, 3 + CANDLE_RANGE_LOOKBACK)
    ) / CANDLE_RANGE_LOOKBACK
    ratio_prev = (prev_range / avg_before_prev) if avg_before_prev > 0 else 0.0

    range_ratio = max(ratio_curr, ratio_prev)
    base["candle_range_ratio"] = round(range_ratio, 2)
    range_cond = ratio_curr >= 2.0 or ratio_prev >= 2.0

    # ── Logique : (① ET (② OU ③))  OU  ④ ────────────────────────────────
    rsi_plus_other = (rsi_bull or rsi_bear) and (imp_bull or imp_bear or ema_bull or ema_bear)

    if not rsi_plus_other and not range_cond:
        base["reject_reason"] = "conditions non remplies"
        return base

    # Construire les listes de signaux
    if rsi_plus_other:
        if rsi_bull:
            bullish_signals.append(f"RSI {rsi:.1f} > {RSI_OVERBOUGHT}")
        else:
            bearish_signals.append(f"RSI {rsi:.1f} < {RSI_OVERSOLD}")
        if imp_bull:  bullish_signals.append(f"Impulsion +{signed_impulse/atr:.1f}×ATR")
        if imp_bear:  bearish_signals.append(f"Impulsion {signed_impulse/atr:.1f}×ATR")
        if ema_bull:  bullish_signals.append(f"EMA dist +{ema_dist_signed/atr:.1f}×ATR")
        if ema_bear:  bearish_signals.append(f"EMA dist {ema_dist_signed/atr:.1f}×ATR")

    if range_cond:
        triggered = []
        if ratio_curr >= 2.0: triggered.append(f"act.×{ratio_curr:.2f}")
        if ratio_prev >= 2.0: triggered.append(f"préc.×{ratio_prev:.2f}")
        label = f"Range ({', '.join(triggered)}) moy"
        if ema_dist_signed >= 0:
            bullish_signals.append(label)
        else:
            bearish_signals.append(label)

    # Direction
    if len(bullish_signals) >= len(bearish_signals):
        direction = "bullish"
        signals   = bullish_signals
    else:
        direction = "bearish"
        signals   = bearish_signals

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
) -> None:
    direction  = result["direction"]
    emoji_main = "🔥" if direction == "bullish" else "❄️"
    arrow      = "🔼" if direction == "bullish" else "🔽"
    tv_url_https = f"https://fr.tradingview.com/chart/?symbol={tv_symbol}"

    signals_text = "\n".join(f"✅ `{s}`" for s in result["signals"])
    now_str = _now_paris().strftime("%d/%m/%Y %H:%M")

    caption = (
        f"*New overextension on {pair} {emoji_main}*\n\n"
        f"🕐 `{now_str}`\n"
        f"{arrow} *Direction :* {direction.capitalize()}\n"
        f"💰 *Prix :* `{result['price']}`\n\n"
        f"📡 *Signaux déclenchés :*\n{signals_text}\n\n"
        f"⚡ *Force du signal :*\n{result['strength_bar']}"
    )

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
    log.info(f"  ④ Range seul      : range bougie actuelle ≥ 2.0× moy des {CANDLE_RANGE_LOOKBACK} précédentes")
    log.info(f"  [COOLDOWN]        : {COOLDOWN_HOURS}h par paire/direction/signaux identiques")
    log.info("=" * 60)

    state      = load_alert_state()
    total_sent = 0

    for pair, info in FOREX_PAIRS.items():
        try:
            # ── Fetch H1 ──────────────────────────────────────────────────
            df = fetch_h1_data(info["yf"])
            if df is None:
                log.info(f"  {pair:<12} | ⚠️  données indisponibles")
                continue

            result = detect_overextension(df)

            # ── Log détaillé des indicateurs ──────────────────────────────
            ok  = "✅"
            nok = "❌"
            rsi_ok = ok if abs(result["rsi"] - 50) >= (50 - RSI_OVERSOLD) else nok
            imp_ok = ok if abs(result["impulse_atr"]) >= ATR_MULT_IMPULSE  else nok
            ema_ok = ok if abs(result["ema_dist_atr"]) >= ATR_MULT_EMA_DIST else nok
            rng_ok = ok if result["candle_range_ratio"] >= 2.0              else nok
            log.info(
                f"  {pair:<12} | "
                f"Prix={result['price']}  ATR={result['atr']}  "
                f"RSI={result['rsi']}{rsi_ok}  "
                f"Imp={result['impulse_atr']:+.2f}×{imp_ok}  "
                f"EMA={result['ema_dist_atr']:+.2f}×{ema_ok}  "
                f"Range=×{result['candle_range_ratio']}{rng_ok}"
            )

            if result["detected"]:
                direction = result["direction"]
                log.info(
                    f"  {pair:<12} | 🚨 OVEREXTENSION {direction.upper()} "
                    f"— {result['strength_bar']} "
                    f"— signaux : {', '.join(result['signals'])}"
                )
                on_cd = is_on_cooldown(state, pair, direction, result["signals"])
                if not on_cd:
                    chart_bytes = generate_chart(df, pair, direction)
                    await send_alert(bot, pair, result, info["tv"], chart_bytes)
                    mark_alerted(state, pair, direction, result["signals"])
                    save_alert_state(state)
                    total_sent += 1
                    log.info(f"  {pair:<12} | ✅ alerte envoyée")
                    await asyncio.sleep(1.5)
                else:
                    log.info(f"  {pair:<12} | 🔒 cooldown actif — rien envoyé")
            else:
                log.info(f"  {pair:<12} | ⛔ non déclenché — {result['reject_reason']}")

        except Exception as exc:
            log.error(f"  {pair:<12} | 💥 erreur inattendue : {exc}", exc_info=True)

    # ── Séparateur de fin de salve épinglé ───────────────────────────────
    if total_sent > 0:
        msg = await bot.send_message(
            chat_id=TELEGRAM_CHANNEL_ID,
            text="‼️" * 15,
        )
        try:
            await bot.pin_chat_message(
                chat_id=TELEGRAM_CHANNEL_ID,
                message_id=msg.message_id,
                disable_notification=True,
            )
            log.info("📌 Séparateur épinglé")
        except Exception as e:
            log.warning(f"Impossible d'épingler le message : {e}")

    log.info("-" * 60)
    log.info(f"Scan terminé — {total_sent} message(s) envoyé(s)")


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
