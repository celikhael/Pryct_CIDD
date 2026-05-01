# 🚀 QUICK START - Dashboard Médico

## ⚡ Instalación en 3 pasos

### Paso 1: Instalar dependencias
```bash
pip install -r requirements.txt
```

### Paso 2: Ir a la carpeta del proyecto
```bash
cd clasificacion_proyecto
```

### Paso 3: Ejecutar el dashboard
```bash
python dashboard.py
```

## 🌐 Abrir el Dashboard

Copia y pega esto en tu navegador:
```
http://localhost:5000
```

## 📋 Checklist

- ✅ Python 3.7+ instalado
- ✅ Carpeta `clasificacion_proyecto` existe
- ✅ Carpeta `data` con `data.csv`
- ✅ Carpeta `model` con:
  - modelo.pkl
  - X_test.pkl
  - y_test.pkl
- ✅ Archivo `dashboard.py` en la carpeta `clasificacion_proyecto`

## 🎨 Qué verás

Una interfaz futurista con:
- 📊 5 tarjetas de métricas principales
- 🎯 Matriz de confusión interactiva
- ⭐ Top 15 características más importantes
- 📈 Gráfico de importancia (horizontal)
- 📉 Gráfico de distribución (dona)
- 📚 Estadísticas del dataset

## ⚙️ Cambiar Puerto

Si quieres usar otro puerto (ejemplo: 8080):

**En Windows:**
```batch
python dashboard.py --port 8080
```

**En Mac/Linux:**
```bash
python dashboard.py --port 8080
```

O edita el archivo `dashboard.py` línea final:
```python
app.run(debug=True, host='0.0.0.0', port=8080)
```

## 🐛 Problemas Comunes

### Puerto 5000 ya en uso

**Windows:**
```bash
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

**Mac/Linux:**
```bash
lsof -i :5000
kill -9 <PID>
```

### ModuleNotFoundError: flask

```bash
pip install flask
```

### No encuentra los archivos del modelo

Asegúrate de ejecutar desde dentro de `clasificacion_proyecto`:
```bash
cd clasificacion_proyecto
python dashboard.py
```

## 📱 Acceso Remoto

Si quieres acceder desde otra máquina en tu red:

```bash
# Encuentra tu IP local (Windows)
ipconfig

# Encuentra tu IP local (Mac/Linux)
ifconfig
```

Luego accede desde otra máquina:
```
http://TU_IP_LOCAL:5000
```

## 🎯 Características del Dashboard

### Métricas Mostradas
- **Accuracy:** Precisión general del modelo
- **Precision:** Precisión en casos positivos
- **Recall:** Sensibilidad/cobertura
- **F1 Score:** Media harmónica
- **AUC:** Área bajo la curva ROC

### Elementos Interactivos
- Hover sobre tarjetas para ver efectos
- Gráficos interactivos con tooltips
- Actualización automática cada 30 segundos

### Diseño
- Tema oscuro (dark mode)
- Colores neón (azul #00d4ff + rojo #e94560)
- Animaciones suaves
- Responsivo en mobile/tablet/desktop

## 🔄 Actualizar Datos

El dashboard se actualiza automáticamente. Si quieres forzar una actualización:
```
Presiona F5 en el navegador
```

## 🛑 Detener el Dashboard

Presiona en la terminal:
```
CTRL + C
```

## 💡 Tips

1. **Pantalla Completa:** F11
2. **DevTools:** F12
3. **Caché:** Ctrl+Shift+Del
4. **Recargar:** F5
5. **Zoom:** Ctrl+

## 📚 Recursos

- [Documentación Flask](https://flask.palletsprojects.com/)
- [Chart.js](https://www.chartjs.org/)
- [Scikit-learn](https://scikit-learn.org/)

## ✨ Disfrutalo!

Tu dashboard está listo. ¡Abre tu navegador y explora tu modelo! 🚀

---

**¿Necesitas ayuda?**
Revisa `INSTRUCCIONES_DASHBOARD.md` para una guía completa.
