@echo off
REM Script de ejecución del Dashboard para Windows

cls
echo ======================================================================
echo 🏥 DASHBOARD MEDICO - Instalacion Automatica
echo ======================================================================
echo.

REM Verificar si Python está instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python no esta instalado
    echo Descargalo desde: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✅ Python detectado: %PYTHON_VERSION%
echo.

REM Verificar que existe la carpeta correcta
if not exist "clasificacion_proyecto" (
    echo ❌ No se encontro la carpeta 'clasificacion_proyecto'
    echo Asegurate de que estes en el directorio correcto
    pause
    exit /b 1
)

cd clasificacion_proyecto

REM Crear requirements.txt si no existe
if not exist "requirements.txt" (
    echo 📝 Creando requirements.txt...
    (
        echo flask^>=2.0.0
        echo pandas^>=1.3.0
        echo joblib^>=1.0.0
        echo scikit-learn^>=1.0.0
        echo numpy^>=1.21.0
    ) > requirements.txt
    echo ✅ requirements.txt creado
)

REM Instalar dependencias
echo.
echo 📦 Instalando dependencias...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Error al instalar dependencias
    pause
    exit /b 1
)

echo ✅ Dependencias instaladas
echo.

REM Verificar archivos necesarios
echo 🔍 Verificando archivos necesarios...

if not exist "data\data.csv" (
    echo ❌ No se encuentra: data\data.csv
    pause
    exit /b 1
)

if not exist "model\modelo.pkl" (
    echo ❌ No se encuentra: model\modelo.pkl
    pause
    exit /b 1
)

if not exist "model\X_test.pkl" (
    echo ❌ No se encuentra: model\X_test.pkl
    pause
    exit /b 1
)

if not exist "model\y_test.pkl" (
    echo ❌ No se encuentra: model\y_test.pkl
    pause
    exit /b 1
)

if not exist "dashboard.py" (
    echo ❌ No se encuentra: dashboard.py
    pause
    exit /b 1
)

echo ✅ Todos los archivos necesarios existen
echo.

REM Iniciar el dashboard
echo ======================================================================
echo 🚀 INICIANDO DASHBOARD
echo ======================================================================
echo.
echo 📊 El dashboard estara disponible en:
echo    → http://localhost:5000
echo    → http://127.0.0.1:5000
echo.
echo 🌐 Abre tu navegador y dirigete a cualquiera de las URLs anteriores
echo.
echo ⏹️  Para detener el servidor, presiona: CTRL+C
echo.
echo ======================================================================
echo.

python dashboard.py
pause
