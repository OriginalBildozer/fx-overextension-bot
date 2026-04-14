#!/usr/bin/env python3
"""
Wrapper pour exécuter le bot en continu en local (sans GitHub Actions).
Lance forex_bot.py toutes les 15 minutes dans une boucle infinie.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from forex_bot import FOREX_PAIRS, TELEGRAM_BOT_TOKEN, TELEGRAM_CHANNEL_ID, scan_all

from dotenv import load_dotenv
from telegram import Bot

load_dotenv()

SCAN_INTERVAL_MIN = 15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("forex_bot.log"),
    ],
)
log = logging.getLogger(__name__)


async def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN manquant dans .env")
    if not TELEGRAM_CHANNEL_ID:
        raise ValueError("TELEGRAM_CHANNEL_ID manquant dans .env")

    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    me  = await bot.get_me()
    log.info(f"Bot connecté : @{me.username}")
    log.info(f"Channel cible : {TELEGRAM_CHANNEL_ID}")
    log.info(f"Paires surveillées : {len(FOREX_PAIRS)}")
    log.info(f"Scan toutes les {SCAN_INTERVAL_MIN} minutes — Ctrl+C pour arrêter\n")

    while True:
        await scan_all(bot)
        next_scan = datetime.utcnow() + timedelta(minutes=SCAN_INTERVAL_MIN)
        log.info(f"Prochain scan à {next_scan.strftime('%H:%M:%S')} UTC\n")
        await asyncio.sleep(SCAN_INTERVAL_MIN * 60)


if __name__ == "__main__":
    asyncio.run(main())
