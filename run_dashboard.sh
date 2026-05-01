#!/bin/bash
# Script de instalación y ejecución del Dashboard

echo "======================================================================"
echo "🏥 DASHBOARD MÉDICO - Instalación Automática"
echo "======================================================================"
echo ""

# Verificar si Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 no está instalado"
    echo "Descárgalo desde: https://www.python.org/downloads/"
    exit 1
fi

echo "✅ Python3 detectado: $(python3 --version)"
echo ""

# Crear directorio de proyecto si no existe
if [ ! -d "clasificacion_proyecto" ]; then
    echo "⚠️  No se encontró la carpeta 'clasificacion_proyecto'"
    echo "Asegúrate de que estés en el directorio correcto"
    exit 1
fi

cd clasificacion_proyecto

# Crear archivo requirements.txt si no existe
if [ ! -f "requirements.txt" ]; then
    echo "📝 Creando requirements.txt..."
    cat > requirements.txt << EOF
flask>=2.0.0
pandas>=1.3.0
joblib>=1.0.0
scikit-learn>=1.0.0
numpy>=1.21.0
EOF
    echo "✅ requirements.txt creado"
fi

# Instalar dependencias
echo ""
echo "📦 Instalando dependencias..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Error al instalar dependencias"
    exit 1
fi

echo "✅ Dependencias instaladas"
echo ""

# Verificar que existen los archivos necesarios
echo "🔍 Verificando archivos necesarios..."

if [ ! -f "data/data.csv" ]; then
    echo "❌ No se encuentra: data/data.csv"
    exit 1
fi

if [ ! -f "model/modelo.pkl" ]; then
    echo "❌ No se encuentra: model/modelo.pkl"
    exit 1
fi

if [ ! -f "model/X_test.pkl" ]; then
    echo "❌ No se encuentra: model/X_test.pkl"
    exit 1
fi

if [ ! -f "model/y_test.pkl" ]; then
    echo "❌ No se encuentra: model/y_test.pkl"
    exit 1
fi

if [ ! -f "dashboard.py" ]; then
    echo "❌ No se encuentra: dashboard.py"
    exit 1
fi

echo "✅ Todos los archivos necesarios existen"
echo ""

# Iniciar el dashboard
echo "======================================================================"
echo "🚀 INICIANDO DASHBOARD"
echo "======================================================================"
echo ""
echo "📊 El dashboard estará disponible en:"
echo "   → http://localhost:5000"
echo "   → http://127.0.0.1:5000"
echo ""
echo "🌐 Abre tu navegador y dirígete a cualquiera de las URLs anteriores"
echo ""
echo "⏹️  Para detener el servidor, presiona: CTRL+C"
echo ""
echo "======================================================================"
echo ""

python3 dashboard.py
