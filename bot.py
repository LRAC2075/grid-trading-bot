#!/usr/bin/env python3
"""
Bot de trading asíncrono RL (Fase 4 - Producción)
Optimizado para despliegues gratuitos (Render / Cloud / Local).
Incluye servidor HTTP para health checks y prevención de suspensión.
"""

import asyncio
import logging
import os
import signal
import sqlite3
import time
from typing import Optional, List, Dict, Tuple

import aiohttp
from aiohttp import web
import numpy as np
import pandas as pd
import onnxruntime as ort

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rl_bot")

# Variables de entorno
OKX_REST_URL = "https://www.okx.com/api/v5/market/candles"
OKX_WS_URL = os.getenv("OKX_WS_URL", "wss://ws.okx.com:8443/ws/v5/public")
SYMBOL = os.getenv("SYMBOL", "BTC-USDT")
DB_PATH = os.getenv("DB_PATH", "trading_state.db")
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "100.0"))
FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))
RISK_PER_TRADE = 0.02
PORT = int(os.getenv("PORT", 8080))


class Database:
    """Persistencia ligera usando SQLite para sobrevivir reinicios."""
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, isolation_level=None)
        self.cursor = self.conn.cursor()
        self._init_db()

    def _init_db(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS state (
                id INTEGER PRIMARY KEY,
                balance REAL,
                peak_balance REAL
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT,
                price REAL,
                size REAL,
                sl REAL,
                tp REAL
            )
        ''')
        self.cursor.execute("SELECT balance FROM state WHERE id=1")
        if not self.cursor.fetchone():
            self.cursor.execute("INSERT INTO state (id, balance, peak_balance) VALUES (1, ?, ?)", 
                                (INITIAL_BALANCE, INITIAL_BALANCE))

    def get_state(self) -> Tuple[float, float]:
        self.cursor.execute("SELECT balance, peak_balance FROM state WHERE id=1")
        return self.cursor.fetchone()

    def update_state(self, balance: float, peak_balance: float):
        self.cursor.execute("UPDATE state SET balance=?, peak_balance=? WHERE id=1", (balance, peak_balance))

    def get_positions(self) -> List[Dict]:
        self.cursor.execute("SELECT id, type, price, size, sl, tp FROM positions")
        return [{"id": r[0], "type": r[1], "price": r[2], "size": r[3], "sl": r[4], "tp": r[5]} for r in self.cursor.fetchall()]

    def add_position(self, p_type: str, price: float, size: float, sl: float, tp: float):
        self.cursor.execute("INSERT INTO positions (type, price, size, sl, tp) VALUES (?, ?, ?, ?, ?)",
                            (p_type, price, size, sl, tp))
        
    def clear_positions(self):
        self.cursor.execute("DELETE FROM positions")


class DataFetcher:
    """Descarga de velas históricas desde OKX API REST."""
    def __init__(self, symbol: str):
        self.symbol = symbol

    async def fetch_klines(self, session: aiohttp.ClientSession, bar: str, limit: int) -> pd.DataFrame:
        params = {"instId": self.symbol, "bar": bar, "limit": str(limit)}
        try:
            async with session.get(OKX_REST_URL, params=params, timeout=10) as response:
                data = await response.json()
                if data.get("code") != "0":
                    logger.error(f"Error API OKX: {data.get('msg')}")
                    return pd.DataFrame()
                
                df = pd.DataFrame(data["data"], columns=["timestamp", "open", "high", "low", "close", "vol", "volCcy", "volume", "confirm"])
                df = df[["timestamp", "open", "high", "low", "close", "volume"]]
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df.sort_values("timestamp").reset_index(drop=True)
        except Exception as e:
            logger.error(f"Fallo descargando velas {bar}: {e}")
            return pd.DataFrame()


class FeatureExtractor:
    """Generación de las 1170 características para el modelo ONNX."""
    @staticmethod
    def calc_tech_indicators(df: pd.DataFrame, lookback: int) -> np.ndarray:
        n = len(df)
        if n == 0:
            return np.zeros((lookback, 7), dtype=np.float32)
        
        c = df['close'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        v = df['volume'].values.astype(float)

        rsi = np.zeros(n)
        if n > 14:
            deltas = np.diff(c, prepend=c[0])
            gains, losses = np.maximum(deltas, 0), np.maximum(-deltas, 0)
            avg_gain, avg_loss = np.mean(gains[:14]), np.mean(losses[:14])
            for i in range(14, n):
                avg_gain = (avg_gain * 13 + gains[i]) / 14
                avg_loss = (avg_loss * 13 + losses[i]) / 14
                rs = avg_gain / avg_loss if avg_loss != 0 else 0
                rsi[i] = 100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + rs))
        
        atr = np.zeros(n)
        if n > 1:
            tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
            atr[1:] = pd.Series(tr).rolling(14, min_periods=1).mean().values
        atr = atr / (np.max(atr) + 1e-8)

        bb_pctb = np.zeros(n)
        if n >= 20:
            sma = pd.Series(c).rolling(20).mean()
            std = pd.Series(c).rolling(20).std()
            bb_pctb = (c - (sma - 2*std)) / (4*std + 1e-8)
        bb_pctb = np.nan_to_num(bb_pctb, nan=0.5)

        fib_382, fib_618 = np.zeros(n), np.zeros(n)
        if n >= 20:
            high_max = pd.Series(h).rolling(20).max().values
            low_min = pd.Series(l).rolling(20).min().values
            diff = high_max - low_min
            fib_382 = np.where(diff > 0, (high_max - 0.382 * diff) / c, 1.0)
            fib_618 = np.where(diff > 0, (high_max - 0.618 * diff) / c, 1.0)

        momentum = np.zeros(n)
        if n > 10:
            momentum[10:] = c[10:] / c[:-10] - 1.0

        vol_rel = np.ones(n)
        if n >= 20:
            avg_vol = pd.Series(v).rolling(20).mean().values
            vol_rel = v / (avg_vol + 1e-8)

        features = np.column_stack([rsi/100.0, atr, bb_pctb, fib_382, fib_618, np.nan_to_num(momentum), np.nan_to_num(vol_rel)])
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

        c_safe, o_safe = np.where(c > 0, c, 1e-8), np.where(o > 0, o, 1e-8)
        returns = np.diff(c, prepend=c[0]) / c_safe[0]
        log_returns = np.diff(np.log(c_safe), prepend=np.log(c_safe[0]))
        hl_range = (h - l) / c_safe
        co_ratio = c / o_safe
        volume_norm = v / (np.max(v) + 1e-8)
        
        rsi = np.zeros(n)
        if n > 14:
            deltas = np.diff(c, prepend=c[0])
            gains, losses = np.maximum(deltas, 0), np.maximum(-deltas, 0)
            avg_gain, avg_loss = np.mean(gains[:14]), np.mean(losses[:14])
            for i in range(14, n):
                avg_gain = (avg_gain * 13 + gains[i]) / 14
                avg_loss = (avg_loss * 13 + losses[i]) / 14
                rs = avg_gain / avg_loss if avg_loss != 0 else 0
                rsi[i] = 100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + rs))

        volatility = np.full_like(c, np.std(returns[-14:]) if len(returns) >= 14 else np.std(returns))
        features = np.column_stack([returns, log_returns, hl_range, co_ratio, volume_norm, rsi/100.0, volatility, np.zeros_like(returns)])
        features = np.nan_to_num(features, nan=0.0)
        
        if n < lookback:
            features = np.vstack([np.zeros((lookback - n, 8), dtype=np.float32), features])
        return features[-lookback:].astype(np.float32)


class InferenceEngine:
    """Motor de inferencia determinista con normalización VecNormalize."""
    def __init__(self, model_path: str, mean_path: str, var_path: str):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.obs_mean = np.load(mean_path).astype(np.float32)
        self.obs_var = np.load(var_path).astype(np.float32)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs_norm = (obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
        obs_norm = np.clip(obs_norm, -10.0, 10.0).reshape(1, -1).astype(np.float32)
        action = self.session.run(None, {self.input_name: obs_norm})[0]
        return action[0]


class TradingBot:
    def __init__(self, db: Database, fetcher: DataFetcher, engine: InferenceEngine):
        self.db = db
        self.fetcher = fetcher
        self.engine = engine
        self.balance, self.peak_balance = self.db.get_state()
        self.positions = self.db.get_positions()
        
    def decode_action(self, action: np.ndarray, ref_price: float) -> Tuple[float, float, float, float]:
        centro = ref_price * (1.0 + 0.1 * action[0])
        amplitud = ref_price * 0.02 * (action[1] + 1.0)
        lower = max(centro - amplitud, 0.1)
        upper = max(centro + amplitud, lower + 1e-6)
        
        sl_norm = (action[2] + 1.0) / 2.0
        tp_norm = (action[3] + 1.0) / 2.0
        sl_pct = 0.005 + sl_norm * (0.05 - 0.005)
        tp_pct = 0.01 + tp_norm * (0.10 - 0.01)
        if tp_pct < 2 * sl_pct:
            tp_pct = 2 * sl_pct
            
        return lower, upper, sl_pct, tp_pct

    async def strategy_loop(self):
        """Re-evalúa la política cada 15 minutos calculando el nuevo grid."""
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    df_15m = await self.fetcher.fetch_klines(session, "15m", 48)
                    df_1h = await self.fetcher.fetch_klines(session, "1H", 24)
                    df_4h = await self.fetcher.fetch_klines(session, "4H", 6)

                    if df_15m.empty or df_1h.empty or df_4h.empty:
                        logger.warning("Velas insuficientes para evaluar. Reintentando en 30s...")
                        await asyncio.sleep(30)
                        continue

                    obs_parts = []
                    for df, lb in [(df_15m, 48), (df_1h, 24), (df_4h, 6)]:
                        obs_parts.append(FeatureExtractor.calc_price_features(df, lb).flatten())
                        obs_parts.append(FeatureExtractor.calc_tech_indicators(df, lb).flatten())
                    
                    obs = np.concatenate(obs_parts)
                    ref_price = df_15m.iloc[-1]['close']
                    
                    action = self.engine.predict(obs)
                    lower, upper, sl_pct, tp_pct = self.decode_action(action, ref_price)
                    
                    logger.info(f"Red Evaluada | Grid: {lower:.2f} - {upper:.2f} | SL: {sl_pct:.2%} | TP: {tp_pct:.2%}")
                    
                    self.db.clear_positions()
                    self.positions = []
                    grid_prices = np.linspace(lower, upper, 20)
                    
                    for level in grid_prices:
                        if level < ref_price:
                            riesgo = RISK_PER_TRADE * self.balance
                            sl_dist = level * sl_pct
                            size = (riesgo / sl_dist) if sl_dist > 0 else 0
                            if size * level > self.balance * 0.5:
                                size = (self.balance * 0.5) / level
                            
                            self.db.add_position('limit_buy', level, size, level - sl_dist, level + (level * tp_pct))
                    
                    self.positions = self.db.get_positions()
                    await asyncio.sleep(900)

                except Exception as e:
                    logger.error(f"Error en strategy_loop: {e}")
                    await asyncio.sleep(60)

    async def execution_loop(self):
        """Monitorea el stream en tiempo real y ejecuta compras/cierres."""
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(OKX_WS_URL, heartbeat=20) as ws:
                        await ws.send_json({"op": "subscribe", "args": [{"channel": "tickers", "instId": SYMBOL}]})
                        logger.info("Conectado a OKX WebSocket Tickers")
                        
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = msg.json()
                                if 'data' in data and data['data']:
                                    price = float(data['data'][0]['last'])
                                    self.check_positions(price)
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
            except Exception as e:
                logger.error(f"WS Desconectado: {e}. Reconectando en 5s...")
                await asyncio.sleep(5)

    def check_positions(self, current_price: float):
        for pos in self.positions[:]:
            if pos['type'] == 'limit_buy' and current_price <= pos['price']:
                cost = pos['size'] * pos['price'] * (1 + FEE_RATE)
                if self.balance >= cost:
                    self.balance -= cost
                    pos['type'] = 'active_long'
                    self.db.cursor.execute("UPDATE positions SET type='active_long' WHERE id=?", (pos['id'],))
                    logger.info(f"Compra Ejecutada | Precio: {pos['price']:.2f} | Tamaño: {pos['size']:.6f} BTC")

            elif pos['type'] == 'active_long':
                if current_price <= pos['sl'] or current_price >= pos['tp']:
                    exit_price = pos['sl'] if current_price <= pos['sl'] else pos['tp']
                    proceeds = pos['size'] * exit_price * (1 - FEE_RATE)
                    self.balance += proceeds
                    
                    if self.balance > self.peak_balance:
                        self.peak_balance = self.balance
                        
                    self.db.update_state(self.balance, self.peak_balance)
                    self.db.cursor.execute("DELETE FROM positions WHERE id=?", (pos['id'],))
                    self.positions.remove(pos)
                    
                    pnl = proceeds - (pos['size'] * pos['price'])
                    logger.info(f"Posición Cerrada a {exit_price:.2f} | PnL: {pnl:.2f} USDT | Balance: {self.balance:.2f} USDT")


# Servidor Web para Health Checks
async def start_web_server(bot: TradingBot):
    app = web.Application()
    
    async def health_check(request):
        drawdown = ((bot.balance - bot.peak_balance) / bot.peak_balance) if bot.peak_balance > 0 else 0.0
        return web.json_response({
            "status": "online",
            "symbol": SYMBOL,
            "balance_usdt": round(bot.balance, 2),
            "peak_balance": round(bot.peak_balance, 2),
            "drawdown_pct": f"{drawdown:.2%}",
            "active_positions": len([p for p in bot.positions if p['type'] == 'active_long']),
            "pending_orders": len([p for p in bot.positions if p['type'] == 'limit_buy'])
        })

    app.router.add_get("/", health_check)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    logger.info(f"Servidor HTTP iniciado en el puerto {PORT}")


async def main():
    db = Database(DB_PATH)
    fetcher = DataFetcher(SYMBOL)
    engine = InferenceEngine("model.onnx", "obs_mean.npy", "obs_var.npy")
    bot = TradingBot(db, fetcher, engine)
    
    logger.info(f"Iniciando Bot RL | Balance Inicial: {bot.balance:.2f} USDT")
    
    web_task = asyncio.create_task(start_web_server(bot))
    strategy_task = asyncio.create_task(bot.strategy_loop())
    execution_task = asyncio.create_task(bot.execution_loop())
    
    await asyncio.gather(web_task, strategy_task, execution_task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot detenido manualmente.")