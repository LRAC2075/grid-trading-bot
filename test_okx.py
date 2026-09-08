# test_okx.py
"""Test directo de la API de OKX"""

import asyncio
import aiohttp
import time
from datetime import datetime, timezone

async def test_okx():
    async with aiohttp.ClientSession() as session:
        # Test 1: Ticker
        print("Test 1: Ticker")
        async with session.get(
            "https://www.okx.com/api/v5/market/ticker",
            params={"instId": "BTC-USDT"}
        ) as response:
            data = await response.json()
            print(f"  Status: {response.status}")
            print(f"  BTC Precio: ${float(data['data'][0]['last']):,.2f}")
        
        # Test 2: Velas históricas
        print("\nTest 2: Velas históricas")
        now_ms = int(time.time() * 1000)
        print(f"  Timestamp actual: {now_ms}")
        
        async with session.get(
            "https://www.okx.com/api/v5/market/history-candles",
            params={
                "instId": "BTC-USDT",
                "bar": "15m",
                "before": str(now_ms),
                "limit": "5"
            }
        ) as response:
            print(f"  Status: {response.status}")
            data = await response.json()
            print(f"  Código: {data.get('code')}")
            print(f"  Mensaje: {data.get('msg')}")
            
            if data.get("code") == "0":
                velas = data.get("data", [])
                print(f"  Velas recibidas: {len(velas)}")
                if velas:
                    print(f"  Primera vela (más reciente):")
                    print(f"    Timestamp: {velas[0][0]}")
                    print(f"    Precio: ${float(velas[0][4]):,.2f}")
                    print(f"  Última vela (más antigua):")
                    print(f"    Timestamp: {velas[-1][0]}")
                    print(f"    Precio: ${float(velas[-1][4]):,.2f}")
            else:
                print(f"  Error OKX: {data.get('msg')}")
        
        # Test 3: Con parámetros diferentes
        print("\nTest 3: Con parámetros alternativos")
        # Usar 'after' en lugar de 'before'
        async with session.get(
            "https://www.okx.com/api/v5/market/history-candles",
            params={
                "instId": "BTC-USDT",
                "bar": "15m",
                "after": str(now_ms - 900000 * 5),  # 5 velas atrás
                "limit": "5"
            }
        ) as response:
            print(f"  Status: {response.status}")
            data = await response.json()
            print(f"  Código: {data.get('code')}")
            if data.get("code") == "0":
                velas = data.get("data", [])
                print(f"  Velas recibidas: {len(velas)}")
                if velas:
                    print(f"  Primera vela timestamp: {velas[0][0]}")
                    print(f"  Última vela timestamp: {velas[-1][0]}")

asyncio.run(test_okx())