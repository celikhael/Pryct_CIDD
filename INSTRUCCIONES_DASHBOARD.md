# 🏥 Dashboard Web Ultra Moderno - Clasificación Médica

## 🎨 Características del Dashboard

✨ **Diseño Futurista**
- Interfaz oscura con acentos cyberpunk (azul neón #00d4ff + rojo #e94560)
- Animaciones suaves y efectos visuales de lujo
- Tipografía moderna (Space Grotesk + Space Mono)
- Degradados y glassmorphism

✅ **Funcionalidades**
- Métricas en tiempo real (Accuracy, Precision, Recall, F1, AUC)
- Matriz de confusión interactiva
- Top 15 características más importantes con gráficos
- Distribución del dataset (pie chart)
- Estadísticas completas del modelo
- Gráficos interactivos con Chart.js
- Actualización automática cada 30 segundos

## 🚀 Instalación y Uso

### 1️⃣ Prerequisitos

Asegúrate de tener instalado:
```bash
pip install flask pandas joblib scikit-learn numpy
```

### 2️⃣ Archivo de Configuración (requirements.txt)

```
flask>=2.0.0
pandas>=1.3.0
joblib>=1.0.0
scikit-learn>=1.0.0
numpy>=1.21.0
```

**Instalar con:**
```bash
pip install -r requirements.txt
```

### 3️⃣ Estructura del Proyecto

```
clasificacion_proyecto/
├── dashboard.py          ← 🎯 ARCHIVO PRINCIPAL
├── data/
│   └── data.csv
├── model/
│   ├── modelo.pkl
│   ├── X_test.pkl
│   └── y_test.pkl
└── src/
    ├── train.py
    ├── evaluate.py
    └── graphs.py
```

### 4️⃣ Ejecutar el Dashboard

```bash
cd clasificacion_proyecto
python dashboard.py
```

**Verás este mensaje:**
```
======================================================================
🚀 DASHBOARD MÉDICO - INICIANDO
======================================================================

📊 Servidor disponible en: http://localhost:5000
🌐 Abre tu navegador y dirígete a: http://127.0.0.1:5000

Presiona CTRL+C para detener el servidor

======================================================================
```

### 5️⃣ Abrir en tu Navegador

- **Opción 1:** http://localhost:5000
- **Opción 2:** http://127.0.0.1:5000
- **Opción 3:** http://[tu-ip-local]:5000

## 📊 Elementos del Dashboard

### Barra Superior (Header)
- Título: "DIAGNÓSTICO INTELIGENTE"
- Subtítulo: "Sistema de Clasificación Médica Avanzado"
- Icono animado 🏥

### Métricas Principales (5 Cards)
1. **Precisión General (Accuracy)** - % de predicciones correctas
2. **Precisión Positivos (Precision)** - % de positivos correctamente clasificados
3. **Sensibilidad (Recall)** - % de casos positivos detectados
4. **F1 Score** - Media armónica entre precision y recall
5. **AUC-ROC** - Medida de discriminación del modelo

### Matriz de Confusión
- **TN (Verde):** Verdaderos Negativos
- **FP (Rojo):** Falsos Positivos
- **FN (Rojo):** Falsos Negativos
- **TP (Verde):** Verdaderos Positivos

### Características Principales
- Top 15 características más importantes
- Barras de progreso animadas
- Porcentaje de importancia
- Animaciones escalonadas

### Gráficos Interactivos
1. **Importancia de Características** - Gráfico horizontal
2. **Distribución del Dataset** - Gráfico de dona

### Estadísticas del Dataset
- Muestras totales
- Número de características
- Casos positivos (M)
- Casos negativos (B)
- Muestras de entrenamiento
- Muestras de prueba

## 🎯 Acciones Interactivas

### Hover Effects
- Las tarjetas se elevan y brillan
- Cambios de color en elementos
- Efectos visuales suave

### Gráficos
- Hover para ver detalles
- Interactividad con Chart.js
- Tooltips personalizados

### Responsive Design
- Funciona en desktop (1600px)
- Funciona en tablet (768px)
- Funciona en mobile (320px)

## 🔧 Personalización

### Cambiar Puerto
En el último archivo `dashboard.py`:
```python
app.run(debug=True, host='0.0.0.0', port=8080)  # Cambiar 5000 a 8080
```

### Cambiar Colores
En el `<style>`:
```css
--color-highlight: #e94560;  /* Rojo */
--color-success: #00d4ff;    /* Azul neón */
--color-warning: #ffd700;    /* Oro */
```

### Cambiar Frecuencia de Actualización
En el JavaScript:
```javascript
setInterval(loadData, 30000);  // Cambiar 30000 ms (30s)
```

## 📱 Características Técnicas

**Frontend:**
- HTML5 Semántico
- CSS3 Avanzado (Grid, Flexbox, Gradientes, Animaciones)
- Vanilla JavaScript (sin dependencias)
- Chart.js para gráficos

**Backend:**
- Flask para servir la aplicación
- joblib para cargar el modelo
- sklearn para métricas

**Performance:**
- Carga inicial: ~2-3 segundos
- Actualización: ~500ms
- Responsive a 60fps

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'flask'"
```bash
pip install flask
```

### Error: "Puerto ya en uso"
```bash
# En Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# En Linux/Mac
lsof -i :5000
kill -9 <PID>
```

### Dashboard no muestra datos
1. Verifica que los archivos del modelo existen:
   - `model/modelo.pkl`
   - `model/X_test.pkl`
   - `model/y_test.pkl`
2. Verifica que `data/data.csv` existe
3. Mira la consola para errores

### Gráficos no se cargan
- Intenta actualizar la página (F5)
- Limpia el caché del navegador (Ctrl+Shift+Del)
- Abre la consola del navegador (F12) para ver errores

## 📖 Documentación de APIs

### GET /api/metrics
Retorna las métricas del modelo:
```json
{
  "accuracy": 0.9649,
  "precision": 0.9756,
  "recall": 0.9302,
  "f1": 0.9524,
  "auc": 0.9953,
  "confusion_matrix": {
    "tn": 63,
    "fp": 3,
    "fn": 8,
    "tp": 100
  }
}
```

### GET /api/features
Retorna características principales:
```json
[
  {
    "feature": "area_worst",
    "importance": 0.1539
  },
  ...
]
```

### GET /api/dataset-info
Retorna información del dataset:
```json
{
  "total_samples": 569,
  "features": 30,
  "positive_cases": 357,
  "negative_cases": 212,
  "test_samples": 114,
  "train_samples": 455
}
```

## 💡 Tips y Trucos

### 1. Ver en Fullscreen
Presiona F11 en el navegador

### 2. Desarrollador
Abre DevTools con F12 para:
- Ver errores en consola
- Inspeccionar elementos
- Verificar network requests

### 3. Exportar Pantalla
- Usa la herramienta de captura (Impr Pant)
- O usa Chrome: Ctrl+Shift+P → "Capture"

### 4. Optimizar Performance
Reduce la frecuencia de actualización:
```javascript
setInterval(loadData, 60000);  // 60 segundos en lugar de 30
```

## 🎓 Recursos Educativos

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Chart.js Documentation](https://www.chartjs.org/)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [CSS Grid & Flexbox](https://css-tricks.com/)

## 📊 Ejemplo de Salida

```
Accuracy:  96.49%
Precision: 97.56%
Recall:    93.02%
F1 Score:  95.24%
AUC:       99.53%

Confusion Matrix:
- TN: 63  (Verdaderos Negativos)
- FP: 3   (Falsos Positivos)
- FN: 8   (Falsos Negativos)
- TP: 100 (Verdaderos Positivos)
```

## 🚀 Deployment en Producción

Para desplegar en un servidor:

```bash
# Instalar gunicorn
pip install gunicorn

# Ejecutar con 4 workers
gunicorn -w 4 dashboard:app

# O con más configuración
gunicorn -w 4 -b 0.0.0.0:8000 dashboard:app
```

## ✨ Características Visuales Destacadas

🎨 **Efectos CSS Avanzados:**
- Gradientes lineales y radiales
- Animaciones fluidas
- Backdrop blur (glassmorphism)
- Sombras dinámicas
- Transformaciones en hover

🔄 **Animaciones:**
- Entrada escalonada de características
- Barras de progreso animadas
- Línea de gradiente en header
- Flotación de icono
- Efectos de shine en hover

📐 **Diseño Responsivo:**
- Mobile first
- Breakpoints en 768px y 1200px
- Grid adaptable
- Fuentes escalables

---

**Creado con ❤️ para Machine Learning**

¿Preguntas? ¡Revisa la documentación o abre un issue!
