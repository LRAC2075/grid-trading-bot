# Script para ejecutar el pipeline
Write-Host "Iniciando Grid Trading Bot - Data Pipeline"
Write-Host "==========================================="

# Activar entorno virtual (En Windows, usa \Scripts\ en lugar de /bin/)
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    .\venv\Scripts\Activate.ps1
} else {
    Write-Host "ERROR: No se encontró el entorno virtual en .\venv\Scripts\Activate.ps1" -ForegroundColor Red
    exit 1
}

# Verificar variables de entorno
if (-Not (Test-Path ".env")) {
    Write-Host "ERROR: Archivo .env no encontrado" -ForegroundColor Red
    exit 1
}

# Leer y exportar variables del archivo .env
Get-Content ".env" | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#")) {
        $parts = $line -split '=', 2
        if ($parts.Length -eq 2) {
            $key = $parts[0].Trim()
            $value = $parts[1].Trim() -replace '^"|"$', '' -replace "^'|'$", ''
            Set-Item -Path "env:\$key" -Value $value
        }
    }
}

# Verificar conexión a Supabase
Write-Host "Verificando conexión a Supabase..."
$pythonScript = @"
import asyncpg
import asyncio
import os
import sys

async def test():
    try:
        conn = await asyncpg.connect(
            host=os.getenv('SUPABASE_URL'),
            port=int(os.getenv('SUPABASE_PORT', '5432')),
            database=os.getenv('SUPABASE_DB', 'postgres'),
            user=os.getenv('SUPABASE_USER', 'postgres'),
            password=os.getenv('SUPABASE_PASSWORD'),
            ssl='require'
        )
        await conn.close()
        print('✓ Conexión exitosa a Supabase')
    except Exception as e:
        print(f'✗ Error de conexión: {e}')
        sys.exit(1)

asyncio.run(test())
"@

# En Windows usualmente se invoca como 'python' en lugar de 'python3'
python -c $pythonScript

# Ejecutar pipeline
Write-Host "Ejecutando pipeline..."
python data_pipeline.py

# Verificar resultado
if ($LASTEXITCODE -eq 0) {
    Write-Host "OK - Pipeline completado exitosamente" -ForegroundColor Green
} else {
    Write-Host "ERROR - Error en el pipeline" -ForegroundColor Red
    exit 1
}