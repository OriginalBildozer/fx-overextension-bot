#!/usr/bin/env python3
"""
Mode simulation — vérifie si les conditions d'overextension étaient
remplies pour une paire donnée à un instant précis.

Usage :
    python3 simulate.py "EUR/USD" "16/04/2026/14/30"
    python3 simulate.py "XAU/USD" "15/04/2026/09/00"
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timezone, timedelta
import pandas as pd
import requests
import yfinance as yf

# Session avec headers navigateur — contourne l'anti-bot Yahoo Finance
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

# ── Import des constantes et fonctions du bot ─────────────────────────────────
from forex_bot import (
    FOREX_PAIRS,
    RSI_OVERBOUGHT, RSI_OVERSOLD, RSI_PERIOD,
    ATR_PERIOD, EMA_FAST,
    ATR_MULT_IMPULSE, ATR_MULT_EMA_DIST,
    IMPULSE_WINDOW, CANDLE_RANGE_LOOKBACK,
    compute_rsi, compute_atr, compute_ema,
    _strength_stars,
)

# ─── Helpers d'affichage ──────────────────────────────────────────────────────
G  = "\033[92m"   # vert
R  = "\033[91m"   # rouge
Y  = "\033[93m"   # jaune
B  = "\033[96m"   # bleu clair
W  = "\033[97m"   # blanc
DIM = "\033[2m"   # grisé
RST = "\033[0m"   # reset

def ok(condition: bool) -> str:
    return f"{G}✅{RST}" if condition else f"{R}❌{RST}"

def line(char="─", n=60):
    print(char * n)

# ─── Fetch historique jusqu'à une date précise ────────────────────────────────

def fetch_until(yf_ticker: str, target_dt: datetime) -> pd.DataFrame | None:
    """Télécharge les 20 derniers jours de données H1 avant target_dt."""
    start = target_dt - timedelta(days=20)
    end   = target_dt + timedelta(hours=2)   # +2h pour être sûr d'inclure target_dt

    try:
        df = yf.download(
            yf_ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d %H:%M:%S"),
            interval="1h",
            progress=False,
            auto_adjust=True,
            session=_YF_SESSION,
        )
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception as e:
        print(f"{R}Erreur fetch : {e}{RST}")
        return None


def slice_at(df: pd.DataFrame, target_dt: datetime) -> pd.DataFrame | None:
    """Retourne le df jusqu'à la bougie la plus proche de target_dt."""
    # Harmoniser timezone
    if df.index.tzinfo is not None and target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=timezone.utc)
    elif df.index.tzinfo is None and target_dt.tzinfo is not None:
        target_dt = target_dt.replace(tzinfo=None)

    sliced = df[df.index <= target_dt]
    if sliced.empty:
        return None
    return sliced


# ─── Simulation détaillée ─────────────────────────────────────────────────────

def simulate(pair: str, target_dt: datetime):
    # Trouver le ticker yfinance
    info = FOREX_PAIRS.get(pair)
    if info is None:
        print(f"{R}Paire inconnue : '{pair}'{RST}")
        print(f"Paires disponibles : {', '.join(FOREX_PAIRS.keys())}")
        return

    line("=")
    print(f"{W}  SIMULATION — {pair}  @  {target_dt.strftime('%d/%m/%Y %H:%M')} UTC{RST}")
    line("=")

    # Fetch + slice
    print(f"{DIM}  Téléchargement des données ({info['yf']})...{RST}")
    df_full = fetch_until(info["yf"], target_dt)
    if df_full is None:
        print(f"{R}  Aucune donnée disponible.{RST}")
        return

    df = slice_at(df_full, target_dt)
    if df is None or len(df) < 30:
        print(f"{R}  Pas assez de données avant cette date ({len(df) if df is not None else 0} bougies).{RST}")
        return

    last_candle_ts = df.index[-1]
    print(f"  Dernière bougie utilisée : {B}{last_candle_ts.strftime('%d/%m/%Y %H:%M')}{RST} UTC")
    print(f"  Nombre de bougies chargées : {len(df)}")
    line()

    # ── Calcul des indicateurs ────────────────────────────────────────────
    df = df.copy()
    df["RSI"]      = compute_rsi(df["Close"], RSI_PERIOD)
    df["ATR"]      = compute_atr(df, ATR_PERIOD)
    df["EMA_fast"] = compute_ema(df["Close"], EMA_FAST)

    last   = df.iloc[-1]
    window = df.iloc[-(IMPULSE_WINDOW + 1):]

    rsi             = float(last["RSI"])
    atr             = float(last["ATR"])
    price           = float(last["Close"])
    ema_fast        = float(last["EMA_fast"])
    impulse_start   = float(window.iloc[0]["Close"])
    signed_impulse  = price - impulse_start
    ema_dist_signed = price - ema_fast

    # Bougie large — range actuel et précédent vs leur moy respective
    current_range   = float(last["High"] - last["Low"])
    avg_before_curr = sum(
        float(df.iloc[-i]["High"] - df.iloc[-i]["Low"])
        for i in range(2, 2 + CANDLE_RANGE_LOOKBACK)
    ) / CANDLE_RANGE_LOOKBACK
    ratio_curr = current_range / avg_before_curr if avg_before_curr > 0 else 0.0

    prev_candle     = df.iloc[-2]
    prev_range      = float(prev_candle["High"] - prev_candle["Low"])
    avg_before_prev = sum(
        float(df.iloc[-i]["High"] - df.iloc[-i]["Low"])
        for i in range(3, 3 + CANDLE_RANGE_LOOKBACK)
    ) / CANDLE_RANGE_LOOKBACK
    ratio_prev = prev_range / avg_before_prev if avg_before_prev > 0 else 0.0

    range_ratio   = max(ratio_curr, ratio_prev)

    print(f"{W}  INDICATEURS{RST}")
    print(f"    Prix           : {B}{price:.5f}{RST}")
    print(f"    ATR ({ATR_PERIOD})        : {atr:.6f}")
    print(f"    RSI ({RSI_PERIOD})        : {rsi:.1f}")
    print(f"    EMA {EMA_FAST}          : {ema_fast:.5f}  (dist brute = {ema_dist_signed:+.5f})")
    print(f"    Impulsion ({IMPULSE_WINDOW}b)  : {signed_impulse:+.5f}  sur {IMPULSE_WINDOW} bougies")
    print(f"    Range actuel   : {current_range:.6f}  (×{ratio_curr:.2f} moy des {CANDLE_RANGE_LOOKBACK} préc.)")
    print(f"    Range préc.    : {prev_range:.6f}  (×{ratio_prev:.2f} moy de ses {CANDLE_RANGE_LOOKBACK} préc.)")
    line()

    # ── Évaluation des conditions ─────────────────────────────────────────
    imp_ratio  = signed_impulse / atr  if atr else 0
    ema_ratio  = ema_dist_signed / atr if atr else 0

    rsi_bull      = rsi > RSI_OVERBOUGHT
    rsi_bear      = rsi < RSI_OVERSOLD
    imp_bull      = imp_ratio >  ATR_MULT_IMPULSE
    imp_bear      = imp_ratio < -ATR_MULT_IMPULSE
    ema_bull      = ema_ratio >  ATR_MULT_EMA_DIST
    ema_bear      = ema_ratio < -ATR_MULT_EMA_DIST
    range_trigger = ratio_curr >= 2.0 or ratio_prev >= 2.0

    # Calcul du gap (combien il manque pour déclencher)
    def gap_rsi_bull():  return f"manque {RSI_OVERBOUGHT - rsi:.1f} pts"
    def gap_rsi_bear():  return f"manque {rsi - RSI_OVERSOLD:.1f} pts"
    def gap_imp():       return f"manque {ATR_MULT_IMPULSE - abs(imp_ratio):.2f}×ATR"
    def gap_ema():       return f"manque {ATR_MULT_EMA_DIST - abs(ema_ratio):.2f}×ATR"
    def gap_range():
        parts = []
        if ratio_curr < 2.0: parts.append(f"act. manque ×{2.0 - ratio_curr:.2f}")
        if ratio_prev < 2.0: parts.append(f"préc. manque ×{2.0 - ratio_prev:.2f}")
        return "  ".join(parts)

    rsi_plus_other = (rsi_bull or rsi_bear) and (imp_bull or imp_bear or ema_bull or ema_bear)

    print(f"{W}  LOGIQUE : (① ET (② OU ③))  OU  ④{RST}")
    print(f"    ① RSI > {RSI_OVERBOUGHT} (haussier) : RSI={rsi:.1f}  {ok(rsi_bull)}"
          + (f"  {DIM}{gap_rsi_bull()}{RST}" if not rsi_bull else ""))
    print(f"       RSI < {RSI_OVERSOLD} (baissier) : RSI={rsi:.1f}  {ok(rsi_bear)}"
          + (f"  {DIM}{gap_rsi_bear()}{RST}" if not rsi_bear else ""))
    print(f"    ② Impulsion > {ATR_MULT_IMPULSE}×ATR   : {imp_ratio:+.2f}×ATR  {ok(imp_bull or imp_bear)}"
          + (f"  {DIM}{gap_imp()}{RST}" if not (imp_bull or imp_bear) else ""))
    print(f"    ③ EMA dist  > {ATR_MULT_EMA_DIST}×ATR   : {ema_ratio:+.2f}×ATR  {ok(ema_bull or ema_bear)}"
          + (f"  {DIM}{gap_ema()}{RST}" if not (ema_bull or ema_bear) else ""))
    print(f"    ④ Range (act.×{ratio_curr:.2f} | préc.×{ratio_prev:.2f}) ≥ 2.0×moy  {ok(range_trigger)}"
          + (f"  {DIM}{gap_range()}{RST}" if not range_trigger else ""))
    line()

    if not rsi_plus_other and not range_trigger:
        print(f"{R}  ✗ Conditions non remplies → pas d'alerte{RST}")
        return

    # ── Résumé direction ──────────────────────────────────────────────────
    bullish_signals = []
    bearish_signals = []
    if rsi_plus_other:
        if rsi_bull:  bullish_signals.append(f"RSI {rsi:.1f}")
        if rsi_bear:  bearish_signals.append(f"RSI {rsi:.1f}")
        if imp_bull:  bullish_signals.append(f"Impulsion +{imp_ratio:.2f}×")
        if imp_bear:  bearish_signals.append(f"Impulsion {imp_ratio:.2f}×")
        if ema_bull:  bullish_signals.append(f"EMA +{ema_ratio:.2f}×")
        if ema_bear:  bearish_signals.append(f"EMA {ema_ratio:.2f}×")
    if range_trigger:
        if ema_dist_signed >= 0:
            bullish_signals.append(f"Range ×{range_ratio:.2f} moy")
        else:
            bearish_signals.append(f"Range ×{range_ratio:.2f} moy")

    direction = "bullish" if len(bullish_signals) >= len(bearish_signals) else "bearish"
    signals   = bullish_signals if direction == "bullish" else bearish_signals

    # ── Verdict final ─────────────────────────────────────────────────────
    strength = len(signals)
    print(f"{G}  🚨 ALERTE DÉCLENCHÉE{RST}")
    print(f"     Direction  : {W}{direction.upper()}{RST}")
    print(f"     Signaux    : {', '.join(signals)}")
    print(f"     Force      : {_strength_stars(strength)}")
    line("=")


# ─── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"{Y}Usage : python3 simulate.py \"EUR/USD\" \"16/04/2026/14/30\"{RST}")
        print(f"        python3 simulate.py \"XAU/USD\" \"15/04/2026/09/00\"")
        sys.exit(1)

    pair_arg = sys.argv[1].upper()
    date_arg = sys.argv[2]

    try:
        target = datetime.strptime(date_arg, "%d/%m/%Y/%H/%M").replace(tzinfo=timezone.utc)
    except ValueError:
        print(f"{R}Format de date invalide. Attendu : jj/mm/yyyy/hh/mm  (ex: 16/04/2026/14/30){RST}")
        sys.exit(1)

    # Accepter "EURUSD" ou "EUR/USD"
    if "/" not in pair_arg:
        pair_arg = pair_arg[:3] + "/" + pair_arg[3:]

    simulate(pair_arg, target)
