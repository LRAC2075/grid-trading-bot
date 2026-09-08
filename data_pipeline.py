#!/usr/bin/env python3
"""
Data Pipeline Final Multi-Timeframe - Grid Trading Bot
Descarga velas históricas de OKX (15m, 1H, 4H) con paginación hacia atrás.
Versión optimizada para Windows y hardware limitado.
"""

import asyncio
import aiohttp
import asyncpg
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import logging
import os
import sys
import signal
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de logging sin emojis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuración central del pipeline"""
    def __init__(self):
        # Supabase
        self.SUPABASE_URL = os.getenv("SUPABASE_URL")
        self.SUPABASE_DB = os.getenv("SUPABASE_DB", "postgres")
        self.SUPABASE_USER = os.getenv("SUPABASE_USER", "postgres")
        self.SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")
        self.SUPABASE_PORT = int(os.getenv("SUPABASE_PORT", "5432"))

        # OKX
        self.OKX_BASE_URL = "https://www.okx.com"
        self.SYMBOL = os.getenv("SYMBOL", "BTC-USDT")
        self.TIMEFRAMES = os.getenv("TIMEFRAMES", "15m,1H,4H").split(',')
        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", "200"))
        self.DAYS_HISTORY = int(os.getenv("DAYS_HISTORY", "200"))  # Cambiado a 200

        # Validación
        if not self.SUPABASE_URL or not self.SUPABASE_PASSWORD:
            raise ValueError("Faltan variables de entorno SUPABASE_URL o SUPABASE_PASSWORD")

class OKXClient:
    """Cliente para interactuar con la API de OKX"""
    def __init__(self, config: Config):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"Content-Type": "application/json"}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_history_candles(self, timeframe: str, after_ts: int, limit: int = 200) -> List[Dict]:
        """
        Descarga velas históricas de OKX usando el parámetro 'after'.
        'after' devuelve velas ANTERIORES a ese timestamp (paginar hacia atrás).
        """
        if not self.session:
            raise RuntimeError("Sesión HTTP no inicializada")

        endpoint = "/api/v5/market/history-candles"
        params = {
            "instId": self.config.SYMBOL,
            "bar": timeframe,
            "after": str(after_ts),
            "limit": str(limit)
        }

        try:
            async with self.session.get(f"{self.config.OKX_BASE_URL}{endpoint}", params=params) as resp:
                if resp.status == 429:
                    retry_after = int(resp.headers.get("Retry-After", "2"))
                    logger.warning(f"Rate limit, esperando {retry_after}s")
                    await asyncio.sleep(retry_after)
                    return []

                resp.raise_for_status()
                data = await resp.json()

                if data.get("code") != "0":
                    logger.error(f"Error OKX: {data.get('msg')}")
                    return []

                raw_klines = data.get("data", [])
                if not raw_klines:
                    return []

                # Procesar velas (orden descendente)
                processed = []
                tf_ms = self._timeframe_ms(timeframe)
                for k in raw_klines:
                    try:
                        ts = int(k[0])
                        open_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                        close_time = open_time + timedelta(milliseconds=tf_ms - 1)
                        processed.append({
                            "symbol": self.config.SYMBOL,
                            "timeframe": timeframe,
                            "open_time": open_time,
                            "close_time": close_time,
                            "open_price": float(k[1]),
                            "high_price": float(k[2]),
                            "low_price": float(k[3]),
                            "close_price": float(k[4]),
                            "volume_base": float(k[5]),
                            "volume_quote": float(k[6]),
                            "trades_count": int(k[8]) if len(k) > 8 else None
                        })
                    except (ValueError, IndexError) as e:
                        logger.error(f"Error procesando vela: {e}")
                return processed

        except Exception as e:
            logger.error(f"Error en fetch_history_candles: {e}")
            return []

    def _timeframe_ms(self, timeframe: str) -> int:
        """Duración del timeframe en milisegundos"""
        mapping = {
            "15m": 900_000,
            "1H": 3_600_000,
            "4H": 14_400_000
        }
        return mapping.get(timeframe, 900_000)

async def save_batch(pool: asyncpg.Pool, klines: List[Dict]) -> int:
    """Guarda un lote de velas en Supabase usando upsert"""
    if not klines:
        return 0

    insert_query = """
    INSERT INTO klines_btc_usdt (
        symbol, timeframe, open_time, close_time,
        open_price, high_price, low_price, close_price,
        volume_base, volume_quote, trades_count
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
    ON CONFLICT (timeframe, open_time) 
    DO UPDATE SET
        high_price = EXCLUDED.high_price,
        low_price = EXCLUDED.low_price,
        close_price = EXCLUDED.close_price,
        volume_base = EXCLUDED.volume_base,
        volume_quote = EXCLUDED.volume_quote
    """

    batch_data = [
        (
            k["symbol"], k["timeframe"], k["open_time"], k["close_time"],
            k["open_price"], k["high_price"], k["low_price"], k["close_price"],
            k["volume_base"], k["volume_quote"], k["trades_count"]
        )
        for k in klines
    ]

    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(insert_query, batch_data)
                return len(batch_data)
    except Exception as e:
        logger.error(f"Error guardando batch: {e}")
        return 0

async def descargar_timeframe(pool: asyncpg.Pool, config: Config, timeframe: str):
    """Descarga un timeframe completo usando paginación hacia atrás"""
    logger.info("=" * 60)
    logger.info(f"DESCARGANDO {config.DAYS_HISTORY} DIAS DE VELAS {timeframe}")
    logger.info("=" * 60)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=config.DAYS_HISTORY)
    end_ms = int(end_time.timestamp() * 1000)
    start_ms = int(start_time.timestamp() * 1000)

    # Comenzamos desde un poco después del presente para incluir la última vela cerrada
    current_after_ms = end_ms + 1000

    total_guardadas = 0
    batch_num = 0

    # Calcular número máximo de batches según timeframe
    tf_ms = OKXClient(config)._timeframe_ms(timeframe)
    velas_por_dia = 24 * 60 * 60 * 1000 // tf_ms
    total_velas_esperadas = config.DAYS_HISTORY * velas_por_dia
    max_batches = total_velas_esperadas // config.BATCH_SIZE + 5  # margen

    async with OKXClient(config) as client:
        while batch_num < max_batches:
            batch_num += 1
            logger.info(
                f"Descargando batch {batch_num}/{max_batches} "
                f"(after={datetime.fromtimestamp(current_after_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}) ..."
            )

            klines = await client.fetch_history_candles(timeframe, current_after_ms, config.BATCH_SIZE)

            if not klines:
                logger.warning("No se recibieron datos, deteniendo")
                break

            guardadas = await save_batch(pool, klines)
            total_guardadas += guardadas

            # La vela más antigua de este lote
            oldest_kline = klines[-1]
            oldest_time = oldest_kline["open_time"]
            oldest_ms = int(oldest_time.timestamp() * 1000)

            logger.info(
                f"  Guardadas {guardadas} velas. Rango: "
                f"{klines[0]['open_time'].strftime('%Y-%m-%d %H:%M')} -> "
                f"{oldest_time.strftime('%Y-%m-%d %H:%M')} (desc)"
            )

            # Si ya llegamos al inicio del periodo, terminamos
            if oldest_ms <= start_ms:
                logger.info("Se alcanzó el inicio del periodo deseado")
                break

            # El siguiente after será justo antes de la vela más antigua obtenida
            current_after_ms = oldest_ms - 1

            await asyncio.sleep(0.3)

    logger.info(f"Descarga completada para {timeframe}: {total_guardadas} velas en {batch_num} batches")

async def main():
    """Función principal: descarga todos los timeframes configurados"""
    config = Config()

    # Conexión a Supabase
    pool = None
    try:
        pool = await asyncpg.create_pool(
            host=config.SUPABASE_URL,
            port=config.SUPABASE_PORT,
            database=config.SUPABASE_DB,
            user=config.SUPABASE_USER,
            password=config.SUPABASE_PASSWORD,
            min_size=2,
            max_size=5,
            command_timeout=60,
            ssl='require'
        )
        logger.info("Conexión a Supabase establecida")
    except Exception as e:
        logger.error(f"Error conectando a Supabase: {e}")
        sys.exit(1)

    try:
        # Descargar cada timeframe secuencialmente para no sobrecargar la API
        for timeframe in config.TIMEFRAMES:
            await descargar_timeframe(pool, config, timeframe)
            # Pausa entre timeframes
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"Error durante la descarga: {e}")
    finally:
        if pool:
            await pool.close()
            logger.info("Conexión a Supabase cerrada")

def signal_handler(sig, frame):
    """Manejar Ctrl+C para salir limpiamente"""
    logger.info("Interrupción recibida, cerrando...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Proceso interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error fatal: {e}")