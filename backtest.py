# backtest_windows.py
"""Backtester sin emojis para Windows"""

import os
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

class GridBacktester:
    """Backtester de Grid Trading estatico"""
    
    def __init__(self, df, num_grids=10, investment=10000, fee_rate=0.001):
        self.df = df
        self.investment = investment
        self.fee_rate = fee_rate
        
        # Calcular rango basado en percentiles
        precio_min = df['low'].quantile(0.05)
        precio_max = df['high'].quantile(0.95)
        
        self.lower = float(precio_min)
        self.upper = float(precio_max)
        self.num_grids = num_grids
        
        # Niveles
        self.levels = np.linspace(self.lower, self.upper, num_grids)
        self.order_quantity = investment / (num_grids * np.mean(self.levels))
        
        # Estado
        self.cash = investment
        self.base_asset = 0.0
        self.pending_orders = {}
        self.trades = []
        self.equity_curve = []
        
        # Arrays
        self.timestamps = df['timestamp'].values
        self.opens = df['open'].astype(float).values
        self.highs = df['high'].astype(float).values
        self.lows = df['low'].astype(float).values
        self.closes = df['close'].astype(float).values
        
        print(f"\nGrid adaptado a datos:")
        print(f"   Rango de precios: ${precio_min:,.2f} - ${precio_max:,.2f}")
        print(f"   Espaciado entre niveles: ${(self.upper - self.lower) / num_grids:,.2f}")
    
    def _inicializar_ordenes(self, precio_actual):
        indices_compras = np.where(self.levels < precio_actual)[0]
        for idx in indices_compras:
            self.pending_orders[int(idx)] = 'buy'
    
    def _ejecutar_orden(self, nivel_idx, tipo, precio, timestamp):
        qty = self.order_quantity
        fee = self.fee_rate * qty * precio
        
        if tipo == 'buy':
            costo_total = qty * precio + fee
            if self.cash >= costo_total:
                self.cash -= costo_total
                self.base_asset += qty
                self.trades.append((timestamp, 'buy', precio, qty, fee))
                del self.pending_orders[nivel_idx]
                
                idx_superior = nivel_idx + 1
                if idx_superior < len(self.levels) and idx_superior not in self.pending_orders:
                    self.pending_orders[idx_superior] = 'sell'
            else:
                del self.pending_orders[nivel_idx]
        else:
            if self.base_asset >= qty:
                ingreso_neto = qty * precio - fee
                self.cash += ingreso_neto
                self.base_asset -= qty
                self.trades.append((timestamp, 'sell', precio, qty, fee))
                del self.pending_orders[nivel_idx]
                
                idx_inferior = nivel_idx - 1
                if idx_inferior >= 0 and idx_inferior not in self.pending_orders:
                    self.pending_orders[idx_inferior] = 'buy'
            else:
                del self.pending_orders[nivel_idx]
    
    def _procesar_barra(self, idx):
        open_, high, low, close = (
            self.opens[idx], self.highs[idx],
            self.lows[idx], self.closes[idx]
        )
        timestamp = self.timestamps[idx]
        
        self.equity_curve.append(self.cash + self.base_asset * open_)
        
        mask = (self.levels >= low) & (self.levels <= high)
        indices_tocados = np.where(mask)[0]
        
        if len(indices_tocados) > 0:
            if close >= open_:
                indices_tocados = np.sort(indices_tocados)
            else:
                indices_tocados = np.sort(indices_tocados)[::-1]
            
            for idx_nivel in indices_tocados:
                idx_nivel = int(idx_nivel)
                if idx_nivel in self.pending_orders:
                    self._ejecutar_orden(
                        idx_nivel,
                        self.pending_orders[idx_nivel],
                        self.levels[idx_nivel],
                        timestamp
                    )
        
        self.equity_curve.append(self.cash + self.base_asset * close)
    
    def run(self):
        if len(self.df) == 0:
            raise ValueError("DataFrame vacio")
        
        self._inicializar_ordenes(self.closes[0])
        
        print(f"\nEjecutando backtest...")
        
        for i in range(len(self.df)):
            self._procesar_barra(i)
        
        self._calcular_metricas()
    
    def _calcular_metricas(self):
        ultimo_precio = self.closes[-1]
        if self.base_asset > 0:
            fee_final = self.fee_rate * self.base_asset * ultimo_precio
            self.cash += self.base_asset * ultimo_precio - fee_final
            self.trades.append((self.timestamps[-1], 'sell_final',
                               ultimo_precio, self.base_asset, fee_final))
            self.base_asset = 0
        
        self.final_equity = self.cash
        self.pnl_neto = self.final_equity - self.investment
        self.retorno = (self.pnl_neto / self.investment) * 100
        
        equity_arr = np.array(self.equity_curve)
        equity_arr = np.append(equity_arr, self.final_equity)
        cummax = np.maximum.accumulate(equity_arr)
        drawdowns = (equity_arr - cummax) / cummax
        self.max_drawdown = drawdowns.min() * 100
        
        self.num_operaciones = len([t for t in self.trades if t[1] in ['buy', 'sell']])
        self.num_compras = len([t for t in self.trades if t[1] == 'buy'])
        self.num_ventas = len([t for t in self.trades if t[1] == 'sell'])
        self.comisiones_totales = sum(t[4] for t in self.trades if t[1] != 'sell_final')
    
    def reporte(self):
        print("\n" + "="*60)
        print("REPORTE DE BACKTEST - GRID TRADING")
        print("="*60)
        print(f"Capital Inicial: ${self.investment:,.2f}")
        print(f"Capital Final: ${self.final_equity:,.2f}")
        print(f"PnL Neto: ${self.pnl_neto:,.2f}")
        print(f"Retorno: {self.retorno:.3f}%")
        print(f"Maximo Drawdown: {self.max_drawdown:.3f}%")
        print("-"*60)
        print(f"Total Operaciones: {self.num_operaciones}")
        print(f"   Compras: {self.num_compras}")
        print(f"   Ventas: {self.num_ventas}")
        print(f"Comisiones Totales: ${self.comisiones_totales:,.2f}")
        print("-"*60)
        print(f"Rango del Grid: ${self.lower:,.0f} - ${self.upper:,.0f}")
        print(f"Niveles: {self.num_grids}")
        print(f"Orden por Nivel: {self.order_quantity:.6f} BTC")
        print("="*60)


def cargar_datos():
    """Carga datos desde Supabase"""
    conn = psycopg2.connect(
        host=os.getenv("SUPABASE_URL"),
        port=int(os.getenv("SUPABASE_PORT", "5432")),
        database=os.getenv("SUPABASE_DB", "postgres"),
        user=os.getenv("SUPABASE_USER", "postgres"),
        password=os.getenv("SUPABASE_PASSWORD"),
        sslmode="require"
    )
    
    query = """
        SELECT 
            open_time as timestamp,
            open_price as open,
            high_price as high,
            low_price as low,
            close_price as close,
            volume_quote as volume
        FROM klines_btc_usdt
        WHERE timeframe = '15m'
        ORDER BY open_time ASC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


if __name__ == "__main__":
    try:
        print("Cargando datos...")
        df = cargar_datos()
        
        print(f"Datos cargados: {len(df):,} velas")
        print(f"Rango: {df['timestamp'].min()} -> {df['timestamp'].max()}")
        
        if len(df) < 100:
            print("\nADVERTENCIA: Pocos datos para backtesting")
            print("Ejecuta primero el pipeline para descargar mas datos")
        
        backtester = GridBacktester(
            df=df,
            num_grids=10,
            investment=10000,
            fee_rate=0.001
        )
        
        backtester.run()
        backtester.reporte()
        
    except Exception as e:
        print(f"Error: {e}")