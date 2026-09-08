import sys
import importlib

def check_module(module_name):
    try:
        importlib.import_module(module_name)
        print(f"OK - {module_name} instalado")
        return True
    except ImportError:
        print(f"ERROR - {module_name} NO instalado")
        return False

def main():
    print("Verificando instalación...")
    print("-" * 40)
    
    modules = ['aiohttp', 'asyncpg', 'backoff', 'dotenv']
    all_ok = True
    
    for module in modules:
        if not check_module(module):
            all_ok = False
            
    print("-" * 40)
    if all_ok:
        print("OK - Todas las dependencias instaladas correctamente")
        print("\nPara ejecutar el pipeline:")
        print(".\\run_pipeline.ps1")
    else:
        print("ERROR - Faltan dependencias. Ejecutar:")
        print("pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()