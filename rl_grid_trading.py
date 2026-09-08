#!/usr/bin/env python3
"""
Fase 3: Entrenamiento RL para Grid Trading Dinámico v3 (corregido)
- Espacio de observación con tamaño fijo (1170).
- Acción 4D: centro, amplitud, SL, TP.
- Gestión de riesgo: posición = (2% balance) / (precio * SL%).
- Bloques de 8h (32 velas 15m), capital entrenamiento 10 USDT.
- Guarda mejor modelo y estadísticas.
"""

import os
import shutil
import time
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any

import psycopg2
from psycopg2 import OperationalError
from dotenv import load_dotenv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

load_dotenv()

# ----------------------------
# 1. Carga de datos
# ----------------------------
def cargar_datos(timeframe: str, dias: int = 200) -> pd.DataFrame:
    max_intentos = 5
    for intento in range(max_intentos):
        try:
            conn = psycopg2.connect(
                host=os.getenv("SUPABASE_URL"),
                port=int(os.getenv("SUPABASE_PORT", "5432")),
                database=os.getenv("SUPABASE_DB", "postgres"),
                user=os.getenv("SUPABASE_USER", "postgres"),
                password=os.getenv("SUPABASE_PASSWORD"),
                sslmode="require",
                connect_timeout=10
            )
            query = f"""
                SELECT open_time, open_price, high_price, low_price, close_price, volume_quote
                FROM klines_btc_usdt
                WHERE timeframe = '{timeframe}'
                AND open_time >= NOW() - INTERVAL '{dias} days'
                ORDER BY open_time ASC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()

            df.rename(columns={
                "open_time": "timestamp", "open_price": "open", "high_price": "high",
                "low_price": "low", "close_price": "close", "volume_quote": "volume"
            }, inplace=True)

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(inplace=True)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)
            print(f"✅ {timeframe}: {len(df)} velas cargadas")
            return df

        except OperationalError as e:
            print(f"Intento {intento+1}/{max_intentos} fallido: {e}")
            if intento == max_intentos - 1:
                raise
            time.sleep(2 ** intento)

# ----------------------------
# 2. Indicadores técnicos
# ----------------------------
def calcular_indicadores(df: pd.DataFrame, lookback: int) -> np.ndarray:
    """
    Calcula indicadores técnicos y devuelve matriz (lookback, 7).
    Rellena con ceros si hay menos de lookback filas para mantener tamaño fijo.
    """
    df = df.sort_values('timestamp').tail(lookback).reset_index(drop=True)
    n = len(df)
    if n == 0:
        return np.zeros((lookback, 7), dtype=np.float32)

    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    close = df['close'].values.astype(float)
    volume = df['volume'].values.astype(float)

    # RSI
    rsi = np.zeros(n)
    period = 14
    if n > period:
        deltas = np.diff(close, prepend=close[0])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        for i in range(period, n):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi / 100.0

    # ATR
    atr = np.zeros(n)
    if n > 1:
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        atr[1:] = pd.Series(tr).rolling(14, min_periods=1).mean().values
    max_atr = np.max(atr)
    atr = atr / max_atr if max_atr > 0 else atr

    # Bollinger %B
    bb_pctb = np.zeros(n)
    if n >= 20:
        sma = pd.Series(close).rolling(20).mean()
        std = pd.Series(close).rolling(20).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        bb_pctb = (close - lower) / (upper - lower + 1e-8)
    bb_pctb = np.nan_to_num(bb_pctb, nan=0.5)

    # Fibonacci
    fib_382 = np.zeros(n)
    fib_618 = np.zeros(n)
    if n >= 20:
        high_max = np.max(high[-20:])
        low_min = np.min(low[-20:])
        diff = high_max - low_min
        if diff > 0:
            fib_382 = (high_max - 0.382 * diff) / close
            fib_618 = (high_max - 0.618 * diff) / close
    fib_382 = np.nan_to_num(fib_382, nan=1.0)
    fib_618 = np.nan_to_num(fib_618, nan=1.0)

    # Momentum
    momentum = np.zeros(n)
    if n > 10:
        momentum[10:] = close[10:] / close[:-10] - 1.0
    momentum = np.nan_to_num(momentum, nan=0.0)

    # Volumen relativo
    vol_rel = np.ones(n)
    if n >= 20:
        avg_vol = pd.Series(volume).rolling(20).mean().values
        vol_rel = volume / (avg_vol + 1e-8)
    vol_rel = np.nan_to_num(vol_rel, nan=1.0)

    features = np.column_stack([rsi, atr, bb_pctb, fib_382, fib_618, momentum, vol_rel])

    # Rellenar con ceros si n < lookback
    if n < lookback:
        padding = np.zeros((lookback - n, 7), dtype=np.float32)
        features = np.vstack([padding, features])

    return features.astype(np.float32)

# ----------------------------
# 3. Entorno
# ----------------------------
class GridTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_15m: pd.DataFrame,
        data_1h: pd.DataFrame,
        data_4h: pd.DataFrame,
        lookback_15m: int = 48,
        lookback_1h: int = 24,
        lookback_4h: int = 6,
        block_size_15m: int = 32,
        initial_balance: float = 10.0,
        fee_rate: float = 0.001,
        num_levels: int = 20,
        reward_scaling: float = 50.0,
        risk_per_trade: float = 0.02,
        min_sl_pct: float = 0.005,
        max_sl_pct: float = 0.05,
        min_tp_pct: float = 0.01,
        max_tp_pct: float = 0.10,
    ):
        super().__init__()
        self.data_15m = data_15m.reset_index(drop=True)
        self.data_1h = data_1h.reset_index(drop=True)
        self.data_4h = data_4h.reset_index(drop=True)
        self.lookback_15m = lookback_15m
        self.lookback_1h = lookback_1h
        self.lookback_4h = lookback_4h
        self.block_size_15m = block_size_15m

        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.num_levels = num_levels
        self.reward_scaling = reward_scaling
        self.risk_per_trade = risk_per_trade
        self.min_sl_pct = min_sl_pct
        self.max_sl_pct = max_sl_pct
        self.min_tp_pct = min_tp_pct
        self.max_tp_pct = max_tp_pct

        self.current_step = 0
        self.balance = initial_balance
        self.equity_curve = [initial_balance]
        self.last_pnl = 0.0

        # Espacios (tamaño fijo)
        self._define_spaces()

    def _define_spaces(self):
        # Acción: 4 dimensiones
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observación: (lookbacks * (8 + 7))
        n_price = 8
        n_tech = 7
        obs_dim = (self.lookback_15m + self.lookback_1h + self.lookback_4h) * (n_price + n_tech)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _extract_price_features(self, df_slice, lookback):
        n_features = 8
        df = df_slice.sort_values('timestamp').tail(lookback).reset_index(drop=True)
        if len(df) == 0:
            return np.zeros((lookback, n_features), dtype=np.float32)

        actual_len = len(df)
        o = df['open'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        c = df['close'].values.astype(float)
        v = df['volume'].values.astype(float)

        with np.errstate(divide='ignore', invalid='ignore'):
            c_safe = np.where(c > 0, c, 1e-8)
            o_safe = np.where(o > 0, o, 1e-8)
            max_v = np.max(v) if np.max(v) > 0 else 1.0
            returns = np.diff(c, prepend=c[0]) / c_safe[0]
            log_returns = np.diff(np.log(c_safe), prepend=np.log(c_safe[0]))
            hl_range = (h - l) / c_safe
            co_ratio = c / o_safe
            volume_norm = v / max_v

            rsi = np.zeros_like(c)
            period = 14
            if len(c) > period:
                deltas = np.diff(c, prepend=c[0])
                gains = np.where(deltas > 0, deltas, 0.0)
                losses = np.where(deltas < 0, -deltas, 0.0)
                avg_gain = np.mean(gains[:period])
                avg_loss = np.mean(losses[:period])
                for i in range(period, len(c)):
                    avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                    avg_loss = (avg_loss * (period - 1) + losses[i]) / period
                    if avg_loss == 0:
                        rsi[i] = 100.0
                    else:
                        rs = avg_gain / avg_loss
                        rsi[i] = 100.0 - 100.0 / (1.0 + rs)

            volatility = np.full_like(c, np.std(returns[-14:]) if len(returns) >= 14 else np.std(returns))

            features = np.column_stack([
                returns, log_returns, hl_range, co_ratio, volume_norm,
                rsi / 100.0, volatility, np.zeros_like(returns)
            ])

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if actual_len < lookback:
            padding = np.zeros((lookback - actual_len, n_features), dtype=np.float32)
            features = np.vstack([padding, features])

        return features.astype(np.float32)

    def _get_observation(self):
        start_idx_15m = self.current_step * self.block_size_15m
        if start_idx_15m >= len(self.data_15m):
            start_idx_15m = len(self.data_15m) - 1
        block_start_time = self.data_15m.iloc[start_idx_15m]['timestamp']

        obs_parts = []
        for df, lb in [
            (self.data_15m, self.lookback_15m),
            (self.data_1h, self.lookback_1h),
            (self.data_4h, self.lookback_4h)
        ]:
            hist = df[df['timestamp'] <= block_start_time]
            price_feats = self._extract_price_features(hist, lb)
            tech_feats = calcular_indicadores(hist, lb)
            obs_parts.append(price_feats.flatten())
            obs_parts.append(tech_feats.flatten())

        obs = np.concatenate(obs_parts).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def _decode_action(self, action):
        idx_15m = self.current_step * self.block_size_15m
        if idx_15m == 0:
            ref_price = self.data_15m.iloc[0]['close']
        else:
            ref_price = self.data_15m.iloc[idx_15m - 1]['close']

        centro = ref_price * (1.0 + 0.1 * action[0])
        amplitud = ref_price * 0.02 * (action[1] + 1.0)
        lower = max(centro - amplitud, 0.1)
        upper = max(centro + amplitud, lower + 1e-6)

        sl_norm = (action[2] + 1.0) / 2.0
        tp_norm = (action[3] + 1.0) / 2.0
        sl_pct = self.min_sl_pct + sl_norm * (self.max_sl_pct - self.min_sl_pct)
        tp_pct = self.min_tp_pct + tp_norm * (self.max_tp_pct - self.min_tp_pct)
        if tp_pct < 2 * sl_pct:
            tp_pct = 2 * sl_pct
        return lower, upper, sl_pct, tp_pct

    def _simulate_grid_with_risk(self, lower, upper, sl_pct, tp_pct):
        start_idx = self.current_step * self.block_size_15m
        end_idx = min(start_idx + self.block_size_15m, len(self.data_15m))
        if start_idx >= end_idx:
            return 0.0

        block_df = self.data_15m.iloc[start_idx:end_idx].reset_index(drop=True)
        initial_price = block_df.iloc[0]['open']
        initial_cash = self.balance
        grid_prices = np.linspace(lower, upper, self.num_levels)

        cash = initial_cash
        btc_holdings = 0.0
        trade_log = []

        for _, row in block_df.iterrows():
            low, high, close = row['low'], row['high'], row['close']
            timestamp = row['timestamp']

            for level_price in grid_prices:
                if low <= level_price <= high:
                    riesgo_total = self.risk_per_trade * initial_cash
                    stop_loss = level_price * sl_pct
                    take_profit = level_price * tp_pct
                    position_size = riesgo_total / stop_loss if stop_loss > 0 else 0.0
                    if position_size * level_price > cash:
                        position_size = cash / level_price
                    if position_size <= 0:
                        continue

                    if level_price < initial_price:
                        cost = position_size * level_price * (1 + self.fee_rate)
                        if cash >= cost:
                            cash -= cost
                            btc_holdings += position_size
                            trade_log.append({
                                'type': 'buy', 'price': level_price, 'size': position_size,
                                'sl': level_price - stop_loss, 'tp': level_price + take_profit,
                                'timestamp': timestamp
                            })
                    else:
                        if btc_holdings >= position_size:
                            proceeds = position_size * level_price * (1 - self.fee_rate)
                            cash += proceeds
                            btc_holdings -= position_size
                            trade_log.append({
                                'type': 'sell', 'price': level_price, 'size': position_size,
                                'sl': level_price + stop_loss, 'tp': level_price - take_profit,
                                'timestamp': timestamp
                            })

            if btc_holdings > 0:
                for trade in trade_log:
                    if trade['type'] == 'buy' and trade['size'] > 0:
                        if low <= trade['sl'] or high >= trade['tp']:
                            exit_price = trade['sl'] if low <= trade['sl'] else trade['tp']
                            proceeds = trade['size'] * exit_price * (1 - self.fee_rate)
                            cash += proceeds
                            btc_holdings -= trade['size']
                            trade['size'] = 0
                            trade_log.append({
                                'type': 'close', 'price': exit_price, 'size': trade['size'],
                                'pnl': proceeds - trade['price'] * trade['size'] * (1 + self.fee_rate)
                            })

        final_price = block_df.iloc[-1]['close']
        if btc_holdings > 0:
            cash += btc_holdings * final_price * (1 - self.fee_rate)

        pnl = cash - initial_cash
        bonus = 0.0
        for t in trade_log:
            if t.get('type') == 'close' and t.get('pnl', 0) > 0:
                if t['pnl'] >= 2 * abs(t.get('price', 0) * t.get('size', 0) * self.risk_per_trade):
                    bonus += 0.01 * self.initial_balance
        return pnl + bonus

    def step(self, action):
        lower, upper, sl_pct, tp_pct = self._decode_action(action)
        pnl = self._simulate_grid_with_risk(lower, upper, sl_pct, tp_pct)
        self.balance += pnl

        self.equity_curve.append(self.balance)
        peak = np.max(self.equity_curve)
        drawdown = (self.balance - peak) / peak if peak > 0 else 0.0

        reward = (pnl / self.initial_balance) * 100 * self.reward_scaling
        reward -= 0.01 * abs(drawdown) * 100

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = self._get_observation()
        info = {
            "pnl": pnl, "balance": self.balance, "lower": lower, "upper": upper,
            "sl_pct": sl_pct, "tp_pct": tp_pct, "drawdown": drawdown
        }
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity_curve = [self.initial_balance]
        self.last_pnl = 0.0
        obs = self._get_observation()
        return obs, {}

    @property
    def max_steps(self):
        return len(self.data_15m) // self.block_size_15m

# ----------------------------
# 4. Preparación de datos y entorno
# ----------------------------
def preparar_datos(dias_historia=200):
    print("Cargando datos...")
    df_15m = cargar_datos("15m", dias_historia)
    df_1h = cargar_datos("1H", dias_historia)
    df_4h = cargar_datos("4H", dias_historia)

    for df in [df_15m, df_1h, df_4h]:
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

    total_blocks = len(df_15m) // 32
    train_end = int(total_blocks * 0.7)
    val_end = int(total_blocks * 0.85)

    train_15m = df_15m.iloc[:train_end*32].copy()
    val_15m = df_15m.iloc[train_end*32:val_end*32].copy()
    test_15m = df_15m.iloc[val_end*32:].copy()

    def filtrar_por_rango(df, start, end):
        return df[(df['timestamp'] >= start) & (df['timestamp'] <= end)].reset_index(drop=True)

    start_train = train_15m.iloc[0]['timestamp']
    end_train = train_15m.iloc[-1]['timestamp']
    start_val = val_15m.iloc[0]['timestamp']
    end_val = val_15m.iloc[-1]['timestamp']
    start_test = test_15m.iloc[0]['timestamp']
    end_test = test_15m.iloc[-1]['timestamp']

    train_1h = filtrar_por_rango(df_1h, start_train, end_train)
    val_1h = filtrar_por_rango(df_1h, start_val, end_val)
    test_1h = filtrar_por_rango(df_1h, start_test, end_test)

    train_4h = filtrar_por_rango(df_4h, start_train, end_train)
    val_4h = filtrar_por_rango(df_4h, start_val, end_val)
    test_4h = filtrar_por_rango(df_4h, start_test, end_test)

    print(f"División: Train={len(train_15m)} velas 15m, Val={len(val_15m)}, Test={len(test_15m)}")
    return (train_15m, train_1h, train_4h), (val_15m, val_1h, val_4h), (test_15m, test_1h, test_4h)

def crear_entorno(data_15m, data_1h, data_4h, capital=10.0):
    env = GridTradingEnv(
        data_15m=data_15m,
        data_1h=data_1h,
        data_4h=data_4h,
        lookback_15m=48,
        lookback_1h=24,
        lookback_4h=6,
        block_size_15m=32,
        initial_balance=capital,
        fee_rate=0.001,
        num_levels=20,
        reward_scaling=50.0,
        risk_per_trade=0.02,
        min_sl_pct=0.005,
        max_sl_pct=0.05,
        min_tp_pct=0.01,
        max_tp_pct=0.10,
    )
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return env

# ----------------------------
# 5. Entrenamiento
# ----------------------------
def entrenar():
    train_data, val_data, test_data = preparar_datos(dias_historia=200)
    train_env = crear_entorno(*train_data, capital=10.0)
    val_env = crear_entorno(*val_data, capital=10.0)

    eval_callback = EvalCallback(
        val_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/eval_results",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-5,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        max_grad_norm=0.5,
        tensorboard_log=None
    )

    print("🚀 Iniciando entrenamiento...")
    model.learn(total_timesteps=800_000, callback=eval_callback)

    best_model_path = "./logs/best_model/best_model.zip"
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, "grid_rl_agent_best.zip")
        print("✅ Mejor modelo copiado a grid_rl_agent_best.zip")
    else:
        model.save("grid_rl_agent_best.zip")
        print("✅ Modelo final guardado como grid_rl_agent_best.zip")

    model.save("grid_rl_agent_final")
    train_env.save("vec_normalize_stats.pkl")
    print("✅ Modelo final y estadísticas guardados")
    return model, test_data

if __name__ == "__main__":
    model, test_data = entrenar()