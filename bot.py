#!/usr/bin/env python3
"""
Bot de trading asíncrono RL (Fase 5 - Monitoreo Visual y Telegram)
Versión final corregida:
- Base de datos asíncrona (aiosqlite)
- Cálculo correcto de retornos y log‑returns
- Manejo seguro de posiciones activas vs. órdenes límite
- Cálculo de PnL con comisiones
- Límite de riesgo total
- Reintentos en fetch de datos
- Control de concurrencia y throttling
- Alertas de Telegram sin bloqueo
"""

import asyncio
import logging
import os
from typing import Optional, List, Dict, Tuple

import aiohttp
from aiohttp import web
import numpy as np
import pandas as pd
import onnxruntime as ort
import aiosqlite

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rl_bot")

# Configuración
OKX_REST_URL = "https://www.okx.com/api/v5/market/candles"
OKX_WS_URL = os.getenv("OKX_WS_URL", "wss://ws.okx.com:8443/ws/v5/public")
SYMBOL = os.getenv("SYMBOL", "BTC-USDT")
DB_PATH = os.getenv("DB_PATH", "trading_state.db")
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "100.0"))
FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))
MAX_TOTAL_RISK = float(os.getenv("MAX_TOTAL_RISK", "0.10"))  # 10% del balance
PORT = int(os.getenv("PORT", "8080"))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

class Database:
    """Clase para manejo asíncrono de SQLite."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[aiosqlite.Connection] = None
        self.lock = asyncio.Lock()

    async def initialize(self):
        self.conn = await aiosqlite.connect(self.db_path)
        self.conn.row_factory = aiosqlite.Row
        await self.conn.execute('''
            CREATE TABLE IF NOT EXISTS state (
                id INTEGER PRIMARY KEY,
                balance REAL,
                peak_balance REAL
            )
        ''')
        await self.conn.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT,
                price REAL,
                size REAL,
                sl REAL,
                tp REAL
            )
        ''')
        await self.conn.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_price REAL,
                exit_price REAL,
                size REAL,
                pnl REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor = await self.conn.execute("SELECT balance FROM state WHERE id=1")
        row = await cursor.fetchone()
        if not row:
            await self.conn.execute(
                "INSERT INTO state (id, balance, peak_balance) VALUES (1, ?, ?)",
                (INITIAL_BALANCE, INITIAL_BALANCE)
            )
        await self.conn.commit()

    async def close(self):
        if self.conn:
            await self.conn.close()

    async def get_state(self) -> Tuple[float, float]:
        async with self.lock:
            cursor = await self.conn.execute("SELECT balance, peak_balance FROM state WHERE id=1")
            row = await cursor.fetchone()
            return row[0], row[1]

    async def update_state(self, balance: float, peak_balance: float):
        async with self.lock:
            await self.conn.execute(
                "UPDATE state SET balance=?, peak_balance=? WHERE id=1",
                (balance, peak_balance)
            )
            await self.conn.commit()

    async def get_positions(self) -> List[Dict]:
        async with self.lock:
            cursor = await self.conn.execute("SELECT id, type, price, size, sl, tp FROM positions")
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def add_position(self, p_type: str, price: float, size: float, sl: float, tp: float):
        async with self.lock:
            await self.conn.execute(
                "INSERT INTO positions (type, price, size, sl, tp) VALUES (?, ?, ?, ?, ?)",
                (p_type, price, size, sl, tp)
            )
            await self.conn.commit()

    async def clear_pending_positions(self):
        """Elimina solo las órdenes límite pendientes (no las activas)."""
        async with self.lock:
            await self.conn.execute("DELETE FROM positions WHERE type='limit_buy'")
            await self.conn.commit()

    async def update_position_type(self, position_id: int, new_type: str):
        async with self.lock:
            await self.conn.execute(
                "UPDATE positions SET type=? WHERE id=?",
                (new_type, position_id)
            )
            await self.conn.commit()

    async def delete_position(self, position_id: int):
        async with self.lock:
            await self.conn.execute("DELETE FROM positions WHERE id=?", (position_id,))
            await self.conn.commit()

    async def position_exists(self, position_id: int, expected_type: str) -> bool:
        async with self.lock:
            cursor = await self.conn.execute(
                "SELECT type FROM positions WHERE id=?", (position_id,)
            )
            row = await cursor.fetchone()
            return row is not None and row[0] == expected_type

    async def log_trade(self, entry_price: float, exit_price: float, size: float, pnl: float):
        async with self.lock:
            await self.conn.execute(
                "INSERT INTO history (entry_price, exit_price, size, pnl) VALUES (?, ?, ?, ?)",
                (entry_price, exit_price, size, pnl)
            )
            await self.conn.commit()

    async def get_summary_stats(self) -> Dict:
        async with self.lock:
            cursor = await self.conn.execute("SELECT pnl FROM history")
            rows = await cursor.fetchall()
            trades = [r[0] for r in rows]
            if not trades:
                return {"total": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "total_pnl": 0.0}
            wins = [t for t in trades if t > 0]
            losses = [t for t in trades if t <= 0]
            return {
                "total": len(trades),
                "win_rate": len(wins) / len(trades),
                "avg_win": float(np.mean(wins)) if wins else 0.0,
                "avg_loss": float(np.mean(losses)) if losses else 0.0,
                "total_pnl": sum(trades)
            }

class TelegramManager:
    def __init__(self, bot_instance, token: Optional[str], chat_id: Optional[str]):
        self.bot = bot_instance
        self.token = token
        self.chat_id = str(chat_id) if chat_id else None
        self.api_url = f"https://api.telegram.org/bot{token}" if token else ""

    async def send_alert(self, text: str):
        if not self.token or not self.chat_id:
            return
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"{self.api_url}/sendMessage",
                    json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
                    timeout=10
                )
        except Exception as e:
            logger.error(f"Error enviando alerta Telegram: {e}")

    def send_alert_background(self, text: str):
        """Lanza la alerta en una tarea independiente sin esperar."""
        asyncio.create_task(self._send_alert_safe(text))

    async def _send_alert_safe(self, text: str):
        try:
            await self.send_alert(text)
        except Exception as e:
            logger.error(f"Error en tarea de alerta: {e}")

    async def handle_webhook(self, request):
        data = await request.json()
        if "message" in data and "text" in data["message"]:
            chat_id = str(data["message"]["chat"]["id"])
            if chat_id != self.chat_id:
                return web.Response()
            text = data["message"]["text"]
            if text.startswith("/status"):
                drawdown = ((self.bot.balance - self.bot.peak_balance) / self.bot.peak_balance) if self.bot.peak_balance > 0 else 0.0
                msg = (
                    f"📊 <b>Estado del Bot</b>\n\n"
                    f"💰 <b>Balance:</b> {self.bot.balance:.2f} USDT\n"
                    f"📉 <b>Drawdown:</b> {drawdown:.2%}\n"
                    f"🛒 <b>Pendientes:</b> {len([p for p in self.bot.positions if p['type'] == 'limit_buy'])}\n"
                    f"⚡ <b>Activas:</b> {len([p for p in self.bot.positions if p['type'] == 'active_long'])}"
                )
                self.send_alert_background(msg)
            elif text.startswith("/positions"):
                actives = [p for p in self.bot.positions if p['type'] == 'active_long']
                if not actives:
                    self.send_alert_background("Sin posiciones activas.")
                else:
                    msg = "⚡ <b>Posiciones Activas:</b>\n\n"
                    for p in actives:
                        msg += f"Entrada: {p['price']:.2f} | Size: {p['size']:.6f}\nSL: {p['sl']:.2f} | TP: {p['tp']:.2f}\n\n"
                    self.send_alert_background(msg)
            elif text.startswith("/summary"):
                stats = await self.bot.db.get_summary_stats()
                msg = (
                    f"🧠 <b>Métricas para Retraining</b>\n\n"
                    f"Total Operaciones: {stats['total']}\n"
                    f"Tasa de Acierto: {stats['win_rate']:.2%}\n"
                    f"Promedio Ganancia: {stats['avg_win']:.2f} USDT\n"
                    f"Promedio Pérdida: {stats['avg_loss']:.2f} USDT\n"
                    f"PnL Histórico: {stats['total_pnl']:.2f} USDT"
                )
                self.send_alert_background(msg)
        return web.Response()

class DataFetcher:
    def __init__(self, symbol: str):
        self.symbol = symbol

    async def fetch_klines(self, session: aiohttp.ClientSession, bar: str, limit: int, retries: int = 3) -> pd.DataFrame:
        params = {"instId": self.symbol, "bar": bar, "limit": str(limit)}
        for attempt in range(retries):
            try:
                async with session.get(OKX_REST_URL, params=params, timeout=10) as response:
                    data = await response.json()
                    if data.get("code") != "0":
                        logger.warning(f"OKX API error: {data.get('msg', 'Unknown')}")
                        return pd.DataFrame()
                    df = pd.DataFrame(
                        data["data"],
                        columns=["timestamp", "open", "high", "low", "close", "vol", "volCcy", "volume", "confirm"]
                    )
                    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    return df.sort_values("timestamp").reset_index(drop=True)
            except Exception as e:
                logger.error(f"Error fetching {bar} klines (intento {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
        return pd.DataFrame()

class FeatureExtractor:
    @staticmethod
    def calc_tech_indicators(df: pd.DataFrame, lookback: int) -> np.ndarray:
        n = len(df)
        if n == 0:
            return np.zeros((lookback, 7), dtype=np.float32)
        c = df['close'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        v = df['volume'].values.astype(float)

        # RSI
        rsi = np.zeros(n)
        if n > 14:
            deltas = np.diff(c, prepend=c[0])
            gains = np.maximum(deltas, 0)
            losses = np.maximum(-deltas, 0)
            avg_gain = np.mean(gains[:14])
            avg_loss = np.mean(losses[:14])
            for i in range(14, n):
                avg_gain = (avg_gain * 13 + gains[i]) / 14
                avg_loss = (avg_loss * 13 + losses[i]) / 14
                rs = avg_gain / avg_loss if avg_loss != 0 else 0
                rsi[i] = 100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + rs))

        # ATR (normalizado)
        atr = np.zeros(n)
        if n > 1:
            tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
            atr[1:] = pd.Series(tr).rolling(14, min_periods=1).mean().values
        atr = atr / (np.max(atr) + 1e-8)

        # Bollinger %B
        bb_pctb = np.zeros(n)
        if n >= 20:
            sma = pd.Series(c).rolling(20).mean()
            std = pd.Series(c).rolling(20).std()
            bb_pctb = (c - (sma - 2*std)) / (4*std + 1e-8)
        bb_pctb = np.nan_to_num(bb_pctb, nan=0.5)

        # Fibonacci (niveles relativos al precio)
        fib_382 = np.zeros(n)
        fib_618 = np.zeros(n)
        if n >= 20:
            high_max = pd.Series(h).rolling(20).max().values
            low_min = pd.Series(l).rolling(20).min().values
            diff = high_max - low_min
            fib_382 = np.where(diff > 0, (high_max - 0.382 * diff) / c, 1.0)
            fib_618 = np.where(diff > 0, (high_max - 0.618 * diff) / c, 1.0)

        # Momentum
        momentum = np.zeros(n)
        if n > 10:
            momentum[10:] = c[10:] / c[:-10] - 1.0

        # Volumen relativo
        vol_rel = np.ones(n)
        if n >= 20:
            avg_vol = pd.Series(v).rolling(20).mean().values
            vol_rel = v / (avg_vol + 1e-8)

        features = np.column_stack([
            rsi/100.0,
            atr,
            bb_pctb,
            fib_382,
            fib_618,
            np.nan_to_num(momentum),
            np.nan_to_num(vol_rel)
        ])
        if n < lookback:
            features = np.vstack([np.zeros((lookback - n, 7), dtype=np.float32), features])
        return features[-lookback:].astype(np.float32)

    @staticmethod
    def calc_price_features(df: pd.DataFrame, lookback: int) -> np.ndarray:
        n = len(df)
        if n == 0:
            return np.zeros((lookback, 8), dtype=np.float32)
        c = df['close'].values.astype(float)
        o = df['open'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        v = df['volume'].values.astype(float)

        # Returns correctos (longitud n)
        returns = np.zeros(n)
        if n > 1:
            returns[1:] = np.diff(c) / np.where(c[:-1] != 0, c[:-1], 1e-8)

        # Log-returns (longitud n)
        log_returns = np.zeros(n)
        if n > 1:
            log_returns[1:] = np.diff(np.log(np.where(c > 0, c, 1e-8)))

        # Rango alto-bajo relativo
        hl_range = (h - l) / np.where(c > 0, c, 1e-8)

        # Relación cierre/apertura
        co_ratio = c / np.where(o > 0, o, 1e-8)

        # Volumen normalizado
        volume_norm = v / (np.max(v) + 1e-8)

        # RSI (reutilizado)
        rsi = np.zeros(n)
        if n > 14:
            deltas = np.diff(c, prepend=c[0])
            gains = np.maximum(deltas, 0)
            losses = np.maximum(-deltas, 0)
            avg_gain = np.mean(gains[:14])
            avg_loss = np.mean(losses[:14])
            for i in range(14, n):
                avg_gain = (avg_gain * 13 + gains[i]) / 14
                avg_loss = (avg_loss * 13 + losses[i]) / 14
                rs = avg_gain / avg_loss if avg_loss != 0 else 0
                rsi[i] = 100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + rs))

        # Volatilidad (rolling std de returns, excluyendo primer cero)
        volatility = np.zeros(n)
        if n > 1:
            if n >= 14:
                volatility[1:] = pd.Series(returns[1:]).rolling(14).std().fillna(0).values
            else:
                volatility = np.full(n, np.std(returns[1:]))

        features = np.column_stack([
            returns,
            log_returns,
            hl_range,
            co_ratio,
            volume_norm,
            rsi/100.0,
            volatility,
            np.zeros_like(returns)  # placeholder
        ])
        features = np.nan_to_num(features, nan=0.0)
        if n < lookback:
            features = np.vstack([np.zeros((lookback - n, 8), dtype=np.float32), features])
        return features[-lookback:].astype(np.float32)

class InferenceEngine:
    def __init__(self, model_path: str, mean_path: str, var_path: str):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.obs_mean = np.load(mean_path).astype(np.float32)
        self.obs_var = np.load(var_path).astype(np.float32)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs_norm = (obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
        obs_norm = np.clip(obs_norm, -10.0, 10.0).reshape(1, -1).astype(np.float32)
        return self.session.run(None, {self.input_name: obs_norm})[0][0]

class TradingBot:
    def __init__(self, db: Database, fetcher: DataFetcher, engine: InferenceEngine):
        self.db = db
        self.fetcher = fetcher
        self.engine = engine
        self.balance, self.peak_balance = 0.0, 0.0
        self.positions: List[Dict] = []
        self.telegram: Optional[TelegramManager] = None
        self.last_analysis: Dict = {}
        self.lock = asyncio.Lock()  # Lock compartido para operaciones críticas
        self.last_price: Optional[float] = None

    async def initialize(self):
        self.balance, self.peak_balance = await self.db.get_state()
        self.positions = await self.db.get_positions()

    def decode_action(self, action: np.ndarray, ref_price: float) -> Tuple[float, float, float, float]:
        centro = ref_price * (1.0 + 0.1 * action[0])
        amplitud = ref_price * 0.02 * (action[1] + 1.0)
        lower = max(centro - amplitud, 0.1)
        upper = max(centro + amplitud, lower + 1e-6)

        sl_pct = 0.005 + ((action[2] + 1.0) / 2.0) * (0.05 - 0.005)
        tp_pct = 0.01 + ((action[3] + 1.0) / 2.0) * (0.10 - 0.01)
        if tp_pct < 2 * sl_pct:
            tp_pct = 2 * sl_pct
        return lower, upper, sl_pct, tp_pct

    async def strategy_loop(self):
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    df_15m = await self.fetcher.fetch_klines(session, "15m", 48)
                    df_1h = await self.fetcher.fetch_klines(session, "1H", 24)
                    df_4h = await self.fetcher.fetch_klines(session, "4H", 6)

                    if df_15m.empty or df_1h.empty or df_4h.empty:
                        logger.warning("Datos insuficientes, esperando...")
                        await asyncio.sleep(30)
                        continue

                    # Construir observación
                    obs_parts = []
                    for df, lb in [(df_15m, 48), (df_1h, 24), (df_4h, 6)]:
                        obs_parts.append(FeatureExtractor.calc_price_features(df, lb).flatten())
                        obs_parts.append(FeatureExtractor.calc_tech_indicators(df, lb).flatten())
                    obs = np.concatenate(obs_parts)
                    ref_price = float(df_15m.iloc[-1]['close'])

                    action = self.engine.predict(obs)
                    lower, upper, sl_pct, tp_pct = self.decode_action(action, ref_price)

                    self.last_analysis = {
                        "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "ref_price": round(ref_price, 2),
                        "raw_action": [round(float(x), 4) for x in action],
                        "grid_lower": round(lower, 2),
                        "grid_upper": round(upper, 2),
                        "sl_pct": round(sl_pct * 100, 2),
                        "tp_pct": round(tp_pct * 100, 2)
                    }

                    # Sección crítica: actualización de posiciones
                    async with self.lock:
                        # Mantener solo posiciones activas, limpiar pendientes
                        await self.db.clear_pending_positions()
                        active_positions = [p for p in self.positions if p['type'] == 'active_long']
                        self.positions = active_positions

                        grid_prices = np.linspace(lower, upper, 20)
                        num_levels = len(grid_prices)
                        total_risk = MAX_TOTAL_RISK * self.balance
                        risk_per_level = min(RISK_PER_TRADE * self.balance, total_risk / max(num_levels, 1))

                        for level in grid_prices:
                            if level < ref_price:
                                sl_dist = level * sl_pct
                                if sl_dist <= 0:
                                    continue
                                size = risk_per_level / sl_dist
                                max_size = (self.balance * 0.5) / level
                                if size > max_size:
                                    size = max_size
                                if size * level <= 0:
                                    continue
                                await self.db.add_position(
                                    'limit_buy',
                                    float(level),
                                    float(size),
                                    float(level - sl_dist),
                                    float(level + (level * tp_pct))
                                )

                        # Recargar posiciones desde DB para tener estado consistente
                        self.positions = await self.db.get_positions()

                    logger.info(f"Estrategia actualizada: {len(self.positions)} posiciones totales, "
                                f"{len([p for p in self.positions if p['type']=='limit_buy'])} pendientes, "
                                f"{len([p for p in self.positions if p['type']=='active_long'])} activas")
                    await asyncio.sleep(900)  # 15 minutos

                except Exception as e:
                    logger.error(f"Error en strategy_loop: {e}", exc_info=True)
                    await asyncio.sleep(60)

    async def execution_loop(self):
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(OKX_WS_URL, heartbeat=20) as ws:
                        await ws.send_json({
                            "op": "subscribe",
                            "args": [{"channel": "tickers", "instId": SYMBOL}]
                        })
                        logger.info("WebSocket conectado, suscrito a tickers")
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = msg.json()
                                if 'data' in data and data['data']:
                                    price = float(data['data'][0]['last'])
                                    if self.last_price is not None:
                                        if abs(price - self.last_price) / self.last_price < 0.0001:
                                            continue
                                    self.last_price = price
                                    await self.check_positions(price)
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                logger.warning("WebSocket cerrado o error, reconectando...")
                                break
            except Exception as e:
                logger.error(f"Error en execution_loop: {e}")
                await asyncio.sleep(5)

    async def check_positions(self, current_price: float):
        # Alertas a enviar después de liberar el lock
        alerts = []

        async with self.lock:
            # Recargar posiciones desde DB para evitar desincronización
            self.positions = await self.db.get_positions()

            for pos in self.positions[:]:
                if pos['type'] == 'limit_buy' and current_price <= pos['price']:
                    # Verificar que la posición siga existiendo como limit_buy
                    if not await self.db.position_exists(pos['id'], 'limit_buy'):
                        continue

                    cost = pos['size'] * pos['price'] * (1 + FEE_RATE)
                    if self.balance >= cost:
                        self.balance -= cost
                        if self.balance > self.peak_balance:
                            self.peak_balance = self.balance
                        await self.db.update_state(self.balance, self.peak_balance)
                        await self.db.update_position_type(pos['id'], 'active_long')
                        pos['type'] = 'active_long'
                        alerts.append(
                            f"🟢 <b>Compra Ejecutada</b>\nPrecio: {pos['price']:.2f}\n"
                            f"Size: {pos['size']:.6f}\nBalance: {self.balance:.2f} USDT"
                        )
                elif pos['type'] == 'active_long':
                    if current_price <= pos['sl'] or current_price >= pos['tp']:
                        exit_price = current_price
                        proceeds = pos['size'] * exit_price * (1 - FEE_RATE)
                        entry_cost = pos['size'] * pos['price'] * (1 + FEE_RATE)
                        pnl = proceeds - entry_cost

                        self.balance += proceeds
                        if self.balance > self.peak_balance:
                            self.peak_balance = self.balance

                        await self.db.update_state(self.balance, self.peak_balance)
                        await self.db.log_trade(pos['price'], exit_price, pos['size'], pnl)
                        await self.db.delete_position(pos['id'])
                        if pos in self.positions:
                            self.positions.remove(pos)

                        icon = "✅" if pnl > 0 else "🛑"
                        alerts.append(
                            f"{icon} <b>Operación Cerrada</b>\n\n"
                            f"Entrada: {pos['price']:.2f}\n"
                            f"Salida: {exit_price:.2f}\n"
                            f"PnL: {pnl:.2f} USDT\n"
                            f"Balance: {self.balance:.2f} USDT"
                        )

        # Enviar alertas fuera del lock
        for alert in alerts:
            if self.telegram:
                self.telegram.send_alert_background(alert)

async def start_web_server(bot: TradingBot):
    telegram_mgr = TelegramManager(bot, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
    bot.telegram = telegram_mgr

    app = web.Application()

    async def health_check(request):
        drawdown = ((bot.balance - bot.peak_balance) / bot.peak_balance) if bot.peak_balance > 0 else 0.0
        return web.json_response({
            "status": "online",
            "symbol": SYMBOL,
            "balance_usdt": round(bot.balance, 2),
            "drawdown_pct": f"{drawdown:.2%}",
            "active_positions": len([p for p in bot.positions if p['type'] == 'active_long'])
        })

    async def api_debug(request):
        return web.json_response(bot.last_analysis)

    async def serve_dashboard(request):
        try:
            with open("dashboard.html", "r", encoding="utf-8") as f:
                return web.Response(text=f.read(), content_type='text/html')
        except FileNotFoundError:
            return web.Response(text="Dashboard no encontrado.", status=404)

    app.router.add_get("/", health_check)
    app.router.add_post("/webhook", telegram_mgr.handle_webhook)
    app.router.add_get("/api/debug", api_debug)
    app.router.add_get("/dashboard", serve_dashboard)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    logger.info(f"Servidor HTTP y Webhook iniciados en puerto {PORT}")

async def main():
    db = Database(DB_PATH)
    await db.initialize()
    fetcher = DataFetcher(SYMBOL)
    engine = InferenceEngine("model.onnx", "obs_mean.npy", "obs_var.npy")
    bot = TradingBot(db, fetcher, engine)
    await bot.initialize()

    web_task = asyncio.create_task(start_web_server(bot))
    strategy_task = asyncio.create_task(bot.strategy_loop())
    execution_task = asyncio.create_task(bot.execution_loop())

    try:
        await asyncio.gather(web_task, strategy_task, execution_task)
    except asyncio.CancelledError:
        pass
    finally:
        await db.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot detenido por el usuario")