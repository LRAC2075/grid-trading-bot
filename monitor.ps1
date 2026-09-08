Write-Host "=== Grid Trading Bot - Monitor ==="
Write-Host ""

# Verificar proceso
$process = Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'python3.exe'" | Where-Object { $_.CommandLine -match "data_pipeline.py" }
if ($process) {
    Write-Host "OK - Pipeline en ejecucion" -ForegroundColor Green
} else {
    Write-Host "INFO - Pipeline no esta corriendo (Es normal si el script ya termino)" -ForegroundColor Yellow
}

# Verificar espacio en disco (Unidad C)
Write-Host "`nEspacio en disco (Unidad C):"
Get-Volume -DriveLetter C | Format-Table | Out-String | Write-Host -NoNewline

# Últimas líneas del log
Write-Host "Ultimas entradas del log:"
if (Test-Path "logs\pipeline.log") {
    Get-Content logs\pipeline.log -Tail 5 | Out-String | Write-Host -NoNewline
} else {
    Write-Host "El archivo logs\pipeline.log no existe aun.`n"
}

# Verificar base de datos
Write-Host "Consultando Supabase..."
$pythonMonitor = @'
import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

async def check_db():
    try:
        conn = await asyncpg.connect(
            host=os.getenv("SUPABASE_URL"),
            port=int(os.getenv("SUPABASE_PORT", "5432")),
            database=os.getenv("SUPABASE_DB", "postgres"),
            user=os.getenv("SUPABASE_USER", "postgres"),
            password=os.getenv("SUPABASE_PASSWORD"),
            ssl="require"
        )
        
        for tf in ["15m", "1H", "4H"]:
            query = f"SELECT COUNT(*) FROM klines_btc_usdt WHERE timeframe = '{tf}'"
            count = await conn.fetchval(query)
            print(f"  {tf}: {count} velas")
            
        await conn.close()
    except Exception as e:
        print(f"Error conectando a BD: {e}")

asyncio.run(check_db())
'@

$pythonMonitor | Out-File -FilePath "temp_monitor.py" -Encoding utf8
python temp_monitor.py
Remove-Item "temp_monitor.py"