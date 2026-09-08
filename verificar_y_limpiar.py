#!/usr/bin/env python3
"""Verifica y limpia datos antiguos, con retención configurable (por defecto 120 días)."""

import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

async def verificar_y_limpiar():
    conn = await asyncpg.connect(
        host=os.getenv("SUPABASE_URL"),
        port=int(os.getenv("SUPABASE_PORT", "5432")),
        database=os.getenv("SUPABASE_DB", "postgres"),
        user=os.getenv("SUPABASE_USER", "postgres"),
        password=os.getenv("SUPABASE_PASSWORD"),
        ssl='require'
    )

    # Configurar días de retención (por defecto 120)
    dias_retencion = int(os.getenv("DIAS_RETENCION", "200"))

    print("=" * 60)
    print(f"ESTADO ACTUAL - Retención: {dias_retencion} días")
    print("=" * 60)

    # Mostrar resumen por timeframe
    for tf in ['15m', '1H', '4H']:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM klines_btc_usdt WHERE timeframe = $1", tf)
        rango = await conn.fetchrow(
            "SELECT MIN(open_time), MAX(open_time) FROM klines_btc_usdt WHERE timeframe = $1", tf)
        print(f"\n{tf}:")
        print(f"  Velas: {count:,}")
        if rango and rango['min']:
            dias = (rango['max'] - rango['min']).days
            print(f"  Rango: {rango['min']} -> {rango['max']} ({dias} días)")

    # Limpiar solo datos más antiguos que la retención
    print(f"\nEliminando velas con más de {dias_retencion} días...")
    for tf in ['15m', '1H', '4H']:
        result = await conn.execute(f"""
            DELETE FROM klines_btc_usdt
            WHERE timeframe = '{tf}'
            AND open_time < NOW() - INTERVAL '{dias_retencion} days'
        """)
        # asyncpg no devuelve el conteo directamente, hacemos una consulta adicional
        count_after = await conn.fetchval(
            "SELECT COUNT(*) FROM klines_btc_usdt WHERE timeframe = $1", tf)
        print(f"  {tf}: {count_after:,} velas restantes")

    await conn.close()
    print("\nLimpieza completada")

if __name__ == "__main__":
    asyncio.run(verificar_y_limpiar())