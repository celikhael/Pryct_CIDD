#!/usr/bin/env python3
"""
Dashboard Web Ultra Moderno para Modelo de Clasificación Médica
Tecnología: Flask + Chart.js + CSS3 Avanzado
"""

from flask import Flask, jsonify, render_template_string, request
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import os
from src.train import train_model

app = Flask(__name__)

# Entrenar (o re-entrenar) el modelo cada vez que arranca el dashboard
print("Entrenando modelo...")
train_model()
print("Modelo listo.")

# ==================== COLUMNAS DE ENTRADA ====================

FEATURE_COLS = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
]

# Ejemplo maligno — ID 842302 (M)
EXAMPLE_MALIGNANT = [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
    0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
    0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
    0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
]

# Ejemplo benigno — ID 857637 (B)
EXAMPLE_BENIGN = [
    9.029, 17.33, 58.79, 250.5, 0.1066, 0.1413, 0.313, 0.04375,
    0.2111, 0.08046, 0.3274, 1.194, 1.885, 17.67, 0.009549, 0.08606,
    0.3038, 0.0646, 0.02675, 0.01737, 9.956, 21.87, 63.62, 292.1,
    0.1696, 0.4244, 0.9454, 0.2112, 0.2882, 0.1155,
]

# ==================== CARGAR MODELO Y DATOS ====================

def load_model_data():
    """Carga el modelo y datos de prueba"""
    try:
        model = joblib.load("model/modelo.pkl")
        X_test = joblib.load("model/X_test.pkl")
        y_test = joblib.load("model/y_test.pkl")
        df = pd.read_csv("data/data.csv")
        
        return {
            'model': model,
            'X_test': X_test,
            'y_test': y_test,
            'df': df
        }
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return None

# Cargar datos
data = load_model_data()

if data:
    model = data['model']
    X_test = data['X_test']
    y_test = data['y_test']
    df = data['df']
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'auc': float(roc_auc_score(y_test, y_pred_proba))
    }
    
    cm = confusion_matrix(y_test, y_pred)
    
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)

# ==================== RUTAS API ====================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/metrics')
def get_metrics():
    return jsonify({
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'auc': metrics['auc'],
        'confusion_matrix': {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
    })

@app.route('/api/features')
def get_features():
    return jsonify([
        {'feature': row['feature'], 'importance': float(row['importance'])}
        for _, row in feature_importance.iterrows()
    ])

@app.route('/api/dataset-info')
def get_dataset_info():
    df_clean = df.drop('id', axis=1)
    return jsonify({
        'total_samples': len(df),
        'features': len(df_clean.columns) - 1,
        'positive_cases': int((df['diagnosis'] == 'M').sum()),
        'negative_cases': int((df['diagnosis'] == 'B').sum()),
        'test_samples': len(X_test),
        'train_samples': len(df) - len(X_test)
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Recibe los atributos del paciente y devuelve la predicción."""
    try:
        body = request.get_json(force=True)
        valores = [float(body[col]) for col in FEATURE_COLS]
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'error': f'Datos de entrada inválidos: {e}'}), 400

    X_input = np.array(valores).reshape(1, -1)

    try:
        prediccion = model.predict(X_input)[0]
        probabilidad = model.predict_proba(X_input)[0]
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    es_maligno = str(prediccion).upper() in ('1', 'M', 'TRUE')
    return jsonify({
        'prediction': 'M' if es_maligno else 'B',
        'label': 'MALIGNO' if es_maligno else 'BENIGNO',
        'prob_malignant': float(probabilidad[1]) if len(probabilidad) > 1 else None,
        'prob_benign': float(probabilidad[0]),
    })

@app.route('/api/example-values')
def example_values():
    return jsonify({
        'malignant': dict(zip(FEATURE_COLS, EXAMPLE_MALIGNANT)),
        'benign':    dict(zip(FEATURE_COLS, EXAMPLE_BENIGN)),
    })

# ==================== TEMPLATE HTML ====================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏥 Dashboard Médico - Clasificación Inteligente</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Space+Mono&display=swap" rel="stylesheet">
    
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        :root {
            --color-primary: #1a1a2e;
            --color-secondary: #16213e;
            --color-accent: #0f3460;
            --color-highlight: #e94560;
            --color-success: #00d4ff;
            --color-warning: #ffd700;
            --color-danger: #e94560;
            --color-text: #ffffff;
            --color-text-secondary: #b0b0b0;
            --font-display: 'Space Grotesk', sans-serif;
            --font-mono: 'Space Mono', monospace;
        }

        body {
            font-family: var(--font-display);
            background: var(--color-primary);
            color: var(--color-text);
            overflow-x: hidden;
        }

        body::before {
            content: '';
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: 
                radial-gradient(circle at 20% 50%, rgba(15,52,96,.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(233,69,96,.1) 0%, transparent 50%);
            pointer-events: none;
            z-index: -1;
        }

        .container { max-width: 1600px; margin: 0 auto; padding: 40px 20px; }

        /* ── HEADER ── */
        .header {
            margin-bottom: 60px;
            padding-bottom: 40px;
            border-bottom: 2px solid rgba(233,69,96,.3);
            position: relative;
            overflow: hidden;
        }
        .header::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #e94560, #00d4ff, #e94560);
            animation: slideRight 3s ease-in-out infinite;
        }
        @keyframes slideRight {
            0%,100% { transform: translateX(-100%); }
            50%      { transform: translateX(100%); }
        }
        .header-content { display: flex; align-items: center; gap: 20px; margin-bottom: 20px; }
        .header-icon { font-size: 48px; animation: float 3s ease-in-out infinite; }
        @keyframes float {
            0%,100% { transform: translateY(0); }
            50%      { transform: translateY(-10px); }
        }
        h1 {
            font-size: 3.5rem; font-weight: 700;
            background: linear-gradient(135deg, #00d4ff, #e94560);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; letter-spacing: -1px;
        }
        .header-subtitle { color: var(--color-text-secondary); font-size: 1.1rem; margin-top: 10px; }

        /* ── METRIC CARDS ── */
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(280px,1fr)); gap: 20px; margin-bottom: 60px; }
        .metric-card {
            background: linear-gradient(135deg, rgba(22,33,62,.8), rgba(15,52,96,.8));
            border: 1px solid rgba(0,212,255,.2);
            border-radius: 16px; padding: 32px;
            position: relative; overflow: hidden;
            transition: all .4s cubic-bezier(.4,0,.2,1);
            cursor: pointer;
        }
        .metric-card::before {
            content: '';
            position: absolute; top: 0; left: -100%; width: 100%; height: 100%;
            background: linear-gradient(90deg, transparent, rgba(0,212,255,.2), transparent);
            transition: left .5s;
        }
        .metric-card:hover { border-color: rgba(0,212,255,.5); transform: translateY(-8px); box-shadow: 0 20px 60px rgba(0,212,255,.15); }
        .metric-card:hover::before { left: 100%; }
        .metric-label { font-size: .95rem; color: var(--color-text-secondary); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 12px; font-weight: 600; font-family: var(--font-mono); }
        .metric-value { font-size: 3.2rem; font-weight: 700; background: linear-gradient(135deg,#00d4ff,#00d4ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-family: var(--font-mono); }
        .metric-card.success .metric-value { background: linear-gradient(135deg,#00ff88,#00d4ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .metric-card.warning .metric-value { background: linear-gradient(135deg,#ffd700,#ffaa00); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .metric-card.danger  .metric-value { background: linear-gradient(135deg,#e94560,#ff6b6b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

        /* ── CONTENT GRID ── */
        .content-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(500px,1fr)); gap: 30px; margin-bottom: 60px; }
        @media (max-width:1200px) { .content-grid { grid-template-columns: 1fr; } }

        .card {
            background: linear-gradient(135deg, rgba(22,33,62,.6), rgba(15,52,96,.6));
            border: 1px solid rgba(233,69,96,.2);
            border-radius: 16px; padding: 32px;
            position: relative; overflow: hidden;
            backdrop-filter: blur(10px);
        }
        .card::before {
            content: '';
            position: absolute; top: -50%; right: -50%;
            width: 200%; height: 200%;
            background: radial-gradient(circle, rgba(0,212,255,.1) 0%, transparent 70%);
            pointer-events: none;
        }
        .card-title {
            font-size: 1.6rem; font-weight: 700; margin-bottom: 24px;
            display: flex; align-items: center; gap: 12px;
            border-bottom: 2px solid rgba(233,69,96,.3); padding-bottom: 16px;
            position: relative; z-index: 1;
        }

        /* ── CONFUSION MATRIX ── */
        .confusion-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; position: relative; z-index: 1; }
        .confusion-cell {
            background: rgba(255,255,255,.05); border: 2px solid rgba(0,212,255,.3);
            border-radius: 12px; padding: 24px; text-align: center;
            transition: all .3s ease; position: relative; overflow: hidden;
        }
        .confusion-cell::before { content: ''; position: absolute; inset: 0; background: radial-gradient(circle at center, rgba(0,212,255,.1), transparent); opacity: 0; transition: opacity .3s; }
        .confusion-cell:hover { border-color: rgba(0,212,255,.7); transform: scale(1.05); }
        .confusion-cell:hover::before { opacity: 1; }
        .confusion-label { font-size: .85rem; color: var(--color-text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; font-weight: 600; font-family: var(--font-mono); }
        .confusion-value { font-size: 2.8rem; font-weight: 700; font-family: var(--font-mono); color: #00d4ff; }
        .confusion-cell.positive .confusion-value { color: #00ff88; }
        .confusion-cell.negative .confusion-value { color: #ffd700; }
        .confusion-cell.error    .confusion-value { color: #e94560; }

        /* ── FEATURES ── */
        .features-list { position: relative; z-index: 1; }
        .feature-item { margin-bottom: 20px; padding-bottom: 20px; border-bottom: 1px solid rgba(233,69,96,.2); animation: fadeInUp .6s ease-out backwards; }
        .feature-item:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
        @keyframes fadeInUp { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }
        .feature-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        .feature-name { font-weight: 600; color: var(--color-text); font-size: .95rem; }
        .feature-percentage { background: rgba(0,212,255,.2); color: #00d4ff; padding: 4px 12px; border-radius: 8px; font-size: .85rem; font-weight: 600; font-family: var(--font-mono); }
        .feature-bar { width: 100%; height: 8px; background: rgba(255,255,255,.1); border-radius: 4px; overflow: hidden; }
        .feature-bar-fill { height: 100%; background: linear-gradient(90deg,#00d4ff,#e94560); border-radius: 4px; animation: expandWidth .8s cubic-bezier(.34,1.56,.64,1) backwards; }
        @keyframes expandWidth { from { width:0; } }

        /* ── CHARTS ── */
        .chart-container { position: relative; height: 350px; margin-bottom: 20px; z-index: 1; }
        canvas { filter: drop-shadow(0 0 20px rgba(0,212,255,.1)); }

        /* ── STATS ── */
        .stats-grid { display: grid; grid-template-columns: repeat(2,1fr); gap: 16px; position: relative; z-index: 1; margin-top: 24px; padding-top: 24px; border-top: 1px solid rgba(233,69,96,.3); }
        .stat-box { background: rgba(0,212,255,.05); border: 1px solid rgba(0,212,255,.2); border-radius: 12px; padding: 16px; text-align: center; transition: all .3s; }
        .stat-box:hover { border-color: rgba(0,212,255,.5); background: rgba(0,212,255,.1); }
        .stat-label { font-size: .8rem; color: var(--color-text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; font-family: var(--font-mono); }
        .stat-value { font-size: 1.8rem; font-weight: 700; color: #00d4ff; font-family: var(--font-mono); }

        /* ── PREDICTION PANEL ── */
        .predict-card {
            background: linear-gradient(135deg, rgba(22,33,62,.85), rgba(26,5,50,.85));
            border: 1px solid rgba(138,43,226,.4);
            border-radius: 16px; padding: 36px;
            position: relative; overflow: hidden;
            backdrop-filter: blur(12px);
            margin-bottom: 60px;
        }
        .predict-card::before {
            content: '';
            position: absolute; top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #8a2be2, #00d4ff, #8a2be2);
        }
        .predict-card .card-title { border-bottom-color: rgba(138,43,226,.4); }

        .predict-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 14px;
            margin-bottom: 28px;
            position: relative; z-index: 1;
        }

        .predict-field label {
            display: block;
            font-size: .78rem;
            color: var(--color-text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
            font-family: var(--font-mono);
        }

        .predict-field input {
            width: 100%;
            background: rgba(255,255,255,.06);
            border: 1px solid rgba(138,43,226,.4);
            border-radius: 8px;
            padding: 9px 12px;
            color: #ffffff;
            font-family: var(--font-mono);
            font-size: .9rem;
            transition: border-color .25s, box-shadow .25s;
            outline: none;
        }
        .predict-field input:focus {
            border-color: #8a2be2;
            box-shadow: 0 0 0 3px rgba(138,43,226,.2);
        }
        .predict-field input.error { border-color: #e94560; box-shadow: 0 0 0 3px rgba(233,69,96,.2); }

        .predict-actions { display: flex; gap: 14px; flex-wrap: wrap; position: relative; z-index: 1; }

        .btn {
            padding: 13px 28px;
            border: none; border-radius: 10px;
            font-family: var(--font-display);
            font-size: 1rem; font-weight: 700;
            cursor: pointer;
            transition: all .3s cubic-bezier(.4,0,.2,1);
            letter-spacing: .5px;
        }
        .btn:hover { transform: translateY(-3px); }
        .btn:active { transform: translateY(0); }

        .btn-predict { background: linear-gradient(135deg,#8a2be2,#6a0dad); color: #fff; box-shadow: 0 6px 24px rgba(138,43,226,.35); }
        .btn-predict:hover { box-shadow: 0 10px 32px rgba(138,43,226,.55); }

        .btn-example { background: rgba(0,212,255,.15); color: #00d4ff; border: 1px solid rgba(0,212,255,.4); }
        .btn-example:hover { background: rgba(0,212,255,.25); }
        .btn-mal { background: rgba(233,69,96,.15); color: #ff6b6b; border: 1px solid rgba(233,69,96,.4); }
        .btn-mal:hover { background: rgba(233,69,96,.28); }
        .btn-ben { background: rgba(0,255,136,.1); color: #00ff88; border: 1px solid rgba(0,255,136,.4); }
        .btn-ben:hover { background: rgba(0,255,136,.2); }

        .btn-clear { background: rgba(233,69,96,.15); color: #e94560; border: 1px solid rgba(233,69,96,.35); }
        .btn-clear:hover { background: rgba(233,69,96,.25); }

        /* ── RESULT BANNER ── */
        #predResult {
            display: none;
            margin-top: 28px;
            border-radius: 14px;
            padding: 28px 32px;
            position: relative;
            z-index: 1;
            animation: fadeInUp .5s ease-out;
        }
        #predResult.malignant {
            background: linear-gradient(135deg, rgba(233,69,96,.15), rgba(180,0,50,.1));
            border: 2px solid rgba(233,69,96,.5);
        }
        #predResult.benign {
            background: linear-gradient(135deg, rgba(0,255,136,.1), rgba(0,180,90,.08));
            border: 2px solid rgba(0,255,136,.4);
        }

        .result-header { display: flex; align-items: center; gap: 18px; margin-bottom: 18px; }
        .result-icon { font-size: 3rem; }
        .result-label { font-size: 2rem; font-weight: 700; font-family: var(--font-mono); }
        #predResult.malignant .result-label { color: #ff6b6b; }
        #predResult.benign    .result-label { color: #00ff88; }

        .prob-bars { display: flex; flex-direction: column; gap: 10px; }
        .prob-row { display: flex; align-items: center; gap: 14px; }
        .prob-name { width: 90px; font-size: .85rem; color: var(--color-text-secondary); font-family: var(--font-mono); text-transform: uppercase; }
        .prob-track { flex: 1; height: 10px; background: rgba(255,255,255,.1); border-radius: 5px; overflow: hidden; }
        .prob-fill { height: 100%; border-radius: 5px; transition: width 1s cubic-bezier(.34,1.56,.64,1); }
        .prob-fill.mal { background: linear-gradient(90deg,#e94560,#ff6b6b); }
        .prob-fill.ben { background: linear-gradient(90deg,#00d4ff,#00ff88); }
        .prob-pct { width: 52px; text-align: right; font-family: var(--font-mono); font-size: .9rem; font-weight: 700; }
        .prob-pct.mal { color: #ff6b6b; }
        .prob-pct.ben { color: #00ff88; }

        /* ── FOOTER ── */
        .footer { text-align: center; padding: 40px 20px; color: var(--color-text-secondary); font-size: .9rem; border-top: 1px solid rgba(233,69,96,.2); margin-top: 60px; }
        .footer p { margin: 8px 0; }

        /* ── SCROLLBAR ── */
        ::-webkit-scrollbar { width: 10px; }
        ::-webkit-scrollbar-track { background: var(--color-primary); }
        ::-webkit-scrollbar-thumb { background: linear-gradient(180deg,#00d4ff,#e94560); border-radius: 5px; }

        /* ── SPINNER ── */
        .spinner {
            display: inline-block;
            width: 20px; height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-top-color: #fff;
            border-radius: 50%;
            animation: spin .7s linear infinite;
            vertical-align: middle;
            margin-right: 8px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        @media (max-width:768px) {
            h1 { font-size: 2.2rem; }
            .metrics-grid, .content-grid { grid-template-columns: 1fr; }
            .predict-grid { grid-template-columns: 1fr 1fr; }
        }
    </style>
</head>
<body>
<div class="container">

    <!-- Header -->
    <div class="header">
        <div class="header-content">
            <div class="header-icon"></div>
            <div>
                <h1>DIAGNÓSTICO INTELIGENTE</h1>
                <p class="header-subtitle">Sistema de Clasificación de Cáncer de Mama — Random Forest</p>
            </div>
        </div>
    </div>

    <!-- Métricas -->
    <div class="metrics-grid" id="metricsContainer">
        <div class="metric-card success"><div class="metric-label">Precisión General</div><div class="metric-value" id="accuracy">0%</div></div>
        <div class="metric-card success"><div class="metric-label">Precisión Positivos</div><div class="metric-value" id="precision">0%</div></div>
        <div class="metric-card success"><div class="metric-label">Sensibilidad</div><div class="metric-value" id="recall">0%</div></div>
        <div class="metric-card warning"><div class="metric-label">F1 Score</div><div class="metric-value" id="f1">0%</div></div>
        <div class="metric-card success"><div class="metric-label">AUC-ROC</div><div class="metric-value" id="auc">0%</div></div>
    </div>

    <!-- Matriz + Features -->
    <div class="content-grid">
        <div class="card">
            <div class="card-title"><span></span> Matriz de Confusión</div>
            <div class="confusion-grid">
                <div class="confusion-cell negative"><div class="confusion-label">Verdaderos Negativos</div><div class="confusion-value" id="tn">0</div></div>
                <div class="confusion-cell error">  <div class="confusion-label">Falsos Positivos</div>    <div class="confusion-value" id="fp">0</div></div>
                <div class="confusion-cell error">  <div class="confusion-label">Falsos Negativos</div>    <div class="confusion-value" id="fn">0</div></div>
                <div class="confusion-cell positive"><div class="confusion-label">Verdaderos Positivos</div><div class="confusion-value" id="tp">0</div></div>
            </div>
        </div>
        <div class="card">
            <div class="card-title"><span></span> Características Principales</div>
            <div class="features-list" id="featuresContainer"><div style="text-align:center;color:var(--color-text-secondary)">Cargando...</div></div>
        </div>
    </div>

    <!-- Gráficos -->
    <div class="content-grid">
        <div class="card">
            <div class="card-title"><span></span> Importancia de Características</div>
            <div class="chart-container"><canvas id="featureChart"></canvas></div>
        </div>
        <div class="card">
            <div class="card-title"><span></span> Distribución de Dataset</div>
            <div class="chart-container"><canvas id="datasetChart"></canvas></div>
        </div>
    </div>

    <!-- ════════════════════════════════════════════════════
         PANEL DE PREDICCIÓN
    ════════════════════════════════════════════════════ -->
    <div class="predict-card">
        <div class="card-title"><span></span> Predicción de Nuevo Caso</div>

        <div class="predict-grid" id="predictGrid">
            <!-- Los campos se generan dinámicamente -->
        </div>

        <div class="predict-actions">
            <button class="btn btn-predict" id="btnPredict" onclick="realizarPrediccion()">
                Predecir Diagnóstico
            </button>
            <button class="btn btn-example btn-mal" onclick="cargarEjemplo(EXAMPLE_MALIGNANT)">
                Ejemplo Maligno
            </button>
            <button class="btn btn-example btn-ben" onclick="cargarEjemplo(EXAMPLE_BENIGN)">
                Ejemplo Benigno
            </button>
            <button class="btn btn-example" onclick="cargarDatosAleatorios()" style="background: rgba(255,215,0,.15); color: #ffd700; border: 1px solid rgba(255,215,0,.4);">
                🎲 Datos Aleatorios
            </button>
            <button class="btn btn-clear" onclick="limpiarCampos()">
                Limpiar Campos
            </button>
        </div>

        <!-- Resultado -->
        <div id="predResult">
            <div class="result-header">
                <div class="result-icon" id="resIcon"></div>
                <div>
                    <div style="font-size:.9rem;color:var(--color-text-secondary);font-family:var(--font-mono);text-transform:uppercase;letter-spacing:2px;margin-bottom:4px">Diagnóstico predicho</div>
                    <div class="result-label" id="resLabel"></div>
                </div>
            </div>
            <div class="prob-bars">
                <div class="prob-row">
                    <div class="prob-name">Maligno</div>
                    <div class="prob-track"><div class="prob-fill mal" id="barMal" style="width:0%"></div></div>
                    <div class="prob-pct mal" id="pctMal">0%</div>
                </div>
                <div class="prob-row">
                    <div class="prob-name">Benigno</div>
                    <div class="prob-track"><div class="prob-fill ben" id="barBen" style="width:0%"></div></div>
                    <div class="prob-pct ben" id="pctBen">0%</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Info dataset -->
    <div class="card">
        <div class="card-title"><span></span> Información del Dataset</div>
        <div class="stats-grid">
            <div class="stat-box"><div class="stat-label">Muestras Totales</div>   <div class="stat-value" id="totalSamples">0</div></div>
            <div class="stat-box"><div class="stat-label">Características</div>    <div class="stat-value" id="numFeatures">0</div></div>
            <div class="stat-box"><div class="stat-label">Casos Positivos</div>    <div class="stat-value" id="positiveCases">0</div></div>
            <div class="stat-box"><div class="stat-label">Casos Negativos</div>    <div class="stat-value" id="negativeCases">0</div></div>
            <div class="stat-box"><div class="stat-label">Muestras Entrenamiento</div><div class="stat-value" id="trainSamples">0</div></div>
            <div class="stat-box"><div class="stat-label">Muestras Prueba</div>    <div class="stat-value" id="testSamples">0</div></div>
        </div>
    </div>

    <div class="footer">
        <p> Powered by Random Forest | Flask + Chart.js | Sistema de Diagnóstico Inteligente</p>
        <p style="margin-top:12px;font-size:.8rem">© 2026 Machine Learning Dashboard</p>
    </div>
</div>

<script>
/* ══════════════════════════════════════════════
   DEFINICIÓN DE CAMPOS
══════════════════════════════════════════════ */
const FEATURE_COLS = [
    "radius_mean","texture_mean","perimeter_mean","area_mean",
    "smoothness_mean","compactness_mean","concavity_mean","concave points_mean",
    "symmetry_mean","fractal_dimension_mean","radius_se","texture_se",
    "perimeter_se","area_se","smoothness_se","compactness_se","concavity_se",
    "concave points_se","symmetry_se","fractal_dimension_se","radius_worst",
    "texture_worst","perimeter_worst","area_worst","smoothness_worst",
    "compactness_worst","concavity_worst","concave points_worst",
    "symmetry_worst","fractal_dimension_worst"
];

const EXAMPLE_MALIGNANT = [
    17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,
    1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,
    25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189
];

const EXAMPLE_BENIGN = [
    9.029,17.33,58.79,250.5,0.1066,0.1413,0.313,0.04375,0.2111,0.08046,
    0.3274,1.194,1.885,17.67,0.009549,0.08606,0.3038,0.0646,0.02675,0.01737,
    9.956,21.87,63.62,292.1,0.1696,0.4244,0.9454,0.2112,0.2882,0.1155
];

/* ── Generar grid de campos de predicción ── */
(function buildGrid() {
    const grid = document.getElementById('predictGrid');
    FEATURE_COLS.forEach((col, i) => {
        const div = document.createElement('div');
        div.className = 'predict-field';
        div.innerHTML = `
            <label for="pf_${i}">${col.replace(/_/g,' ')}</label>
            <input id="pf_${i}" type="number" step="any" placeholder="0.0000"
                   data-col="${col}" value="">
        `;
        grid.appendChild(div);
    });
})();

function getInputs() {
    return FEATURE_COLS.map(col =>
        document.querySelector(`input[data-col="${col}"]`)
    );
}

function cargarDatosAleatorios() {
    // Generar valores aleatorios dentro de rangos realistas basados en ejemplos
    // Rango: valor mínimo y máximo de los ejemplos + 20% de margen
    const rangos = {
        radius_mean: [8, 30],
        texture_mean: [8, 25],
        perimeter_mean: [50, 190],
        area_mean: [200, 2100],
        smoothness_mean: [0.05, 0.15],
        compactness_mean: [0.03, 0.35],
        concavity_mean: [0, 0.43],
        concave_points_mean: [0, 0.3],
        symmetry_mean: [0.1, 0.3],
        fractal_dimension_mean: [0.05, 0.1],
        radius_se: [0.2, 2.9],
        texture_se: [0.3, 4.8],
        perimeter_se: [1, 21],
        area_se: [6, 542],
        smoothness_se: [0.001, 0.03],
        compactness_se: [0.002, 0.14],
        concavity_se: [0, 0.4],
        concave_points_se: [0, 0.05],
        symmetry_se: [0.008, 0.08],
        fractal_dimension_se: [0.001, 0.03],
        radius_worst: [7.5, 36],
        texture_worst: [12, 49],
        perimeter_worst: [50, 251],
        area_worst: [185, 4254],
        smoothness_worst: [0.07, 0.22],
        compactness_worst: [0.03, 1.1],
        concavity_worst: [0, 1.25],
        concave_points_worst: [0, 0.29],
        symmetry_worst: [0.13, 0.66],
        fractal_dimension_worst: [0.055, 0.21]
    };

    const valores = FEATURE_COLS.map(col => {
        const [min, max] = rangos[col] || [0, 1];
        // Generar número aleatorio entre min y max
        return Math.random() * (max - min) + min;
    });

    cargarEjemplo(valores);
}


function cargarEjemplo(valores) {
    getInputs().forEach((inp, i) => {
        inp.value = valores[i];
        inp.classList.remove('error');
    });
    document.getElementById('predResult').style.display = 'none';
}

function limpiarCampos() {
    getInputs().forEach(inp => { inp.value = ''; inp.classList.remove('error'); });
    document.getElementById('predResult').style.display = 'none';
}

async function realizarPrediccion() {
    const inputs = getInputs();
    const body = {};
    let valid = true;

    inputs.forEach((inp, i) => {
        const val = parseFloat(inp.value);
        if (isNaN(val)) {
            inp.classList.add('error');
            valid = false;
        } else {
            inp.classList.remove('error');
            body[FEATURE_COLS[i]] = val;
        }
    });

    if (!valid) {
        alert('Por favor, completa todos los campos con valores numéricos.');
        return;
    }

    const btn = document.getElementById('btnPredict');
    btn.innerHTML = '<span class="spinner"></span> Procesando...';
    btn.disabled = true;

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body)
        });
        const data = await res.json();

        if (data.error) { alert('Error: ' + data.error); return; }

        const isMal  = data.prediction === 'M';
        const pctMal = ((data.prob_malignant ?? 0) * 100);
        const pctBen = ((data.prob_benign   ?? 0) * 100);

        const panel = document.getElementById('predResult');
        panel.className = isMal ? 'malignant' : 'benign';
        panel.style.display = 'block';

        document.getElementById('resIcon').textContent  = isMal ? '' : '';
        document.getElementById('resLabel').textContent = data.label + (isMal ? ' (M)' : ' (B)');

        // Animar barras con un tick de delay para que la transición CSS sea visible
        setTimeout(() => {
            document.getElementById('barMal').style.width = pctMal.toFixed(1) + '%';
            document.getElementById('barBen').style.width = pctBen.toFixed(1) + '%';
        }, 50);

        document.getElementById('pctMal').textContent = pctMal.toFixed(1) + '%';
        document.getElementById('pctBen').textContent = pctBen.toFixed(1) + '%';

        panel.scrollIntoView({behavior:'smooth', block:'nearest'});

    } catch (err) {
        alert('Error de conexión: ' + err.message);
    } finally {
        btn.innerHTML = 'Predecir Diagnóstico';
        btn.disabled = false;
    }
}

/* ══════════════════════════════════════════════
   CARGA DE MÉTRICAS Y GRÁFICOS
══════════════════════════════════════════════ */
let featureChart = null;
let datasetChart = null;

async function loadData() {
    try {
        const metricsRes = await fetch('/api/metrics');
        const metrics = await metricsRes.json();

        document.getElementById('accuracy').textContent = (metrics.accuracy * 100).toFixed(1) + '%';
        document.getElementById('precision').textContent = (metrics.precision * 100).toFixed(1) + '%';
        document.getElementById('recall').textContent   = (metrics.recall    * 100).toFixed(1) + '%';
        document.getElementById('f1').textContent       = (metrics.f1        * 100).toFixed(1) + '%';
        document.getElementById('auc').textContent      = (metrics.auc       * 100).toFixed(1) + '%';

        document.getElementById('tn').textContent = metrics.confusion_matrix.tn;
        document.getElementById('fp').textContent = metrics.confusion_matrix.fp;
        document.getElementById('fn').textContent = metrics.confusion_matrix.fn;
        document.getElementById('tp').textContent = metrics.confusion_matrix.tp;

        const featuresRes = await fetch('/api/features');
        const features = await featuresRes.json();

        document.getElementById('featuresContainer').innerHTML = features.map((f, i) => `
            <div class="feature-item" style="animation-delay:${i * 50}ms">
                <div class="feature-header">
                    <span class="feature-name">${f.feature}</span>
                    <span class="feature-percentage">${(f.importance * 100).toFixed(2)}%</span>
                </div>
                <div class="feature-bar">
                    <div class="feature-bar-fill" style="width:${f.importance * 100}%;animation-delay:${i * 50}ms"></div>
                </div>
            </div>
        `).join('');

        const datasetRes = await fetch('/api/dataset-info');
        const di = await datasetRes.json();

        document.getElementById('totalSamples').textContent = di.total_samples;
        document.getElementById('numFeatures').textContent  = di.features;
        document.getElementById('positiveCases').textContent= di.positive_cases;
        document.getElementById('negativeCases').textContent= di.negative_cases;
        document.getElementById('trainSamples').textContent = di.train_samples;
        document.getElementById('testSamples').textContent  = di.test_samples;

        /* Chart de importancia */
        if (features.length > 0) {
            const ctx = document.getElementById('featureChart').getContext('2d');
            if (featureChart) featureChart.destroy();
            featureChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: features.map(f => f.feature),
                    datasets: [{
                        label: 'Importancia',
                        data: features.map(f => (f.importance * 100).toFixed(2)),
                        backgroundColor: features.map((_,i) => ['#00d4ff','#00ff88','#ffd700','#e94560'][i % 4]),
                        borderColor: 'rgba(0,212,255,.3)',
                        borderWidth: 2,
                        borderRadius: 8,
                        hoverBackgroundColor: '#e94560'
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false, indexAxis: 'y',
                    plugins: {
                        legend: { display: false },
                        tooltip: { backgroundColor:'rgba(26,26,46,.9)', titleColor:'#00d4ff', bodyColor:'#fff', borderColor:'rgba(0,212,255,.5)', borderWidth:1, padding:12 }
                    },
                    scales: {
                        x: { grid:{color:'rgba(255,255,255,.1)'}, ticks:{color:'rgba(255,255,255,.7)'}, max:20 },
                        y: { grid:{display:false},                ticks:{color:'rgba(255,255,255,.7)'} }
                    }
                }
            });
        }

        /* Chart de distribución */
        const ctx2 = document.getElementById('datasetChart').getContext('2d');
        if (datasetChart) datasetChart.destroy();
        datasetChart = new Chart(ctx2, {
            type: 'doughnut',
            data: {
                labels: ['Positivos (M)', 'Negativos (B)'],
                datasets: [{
                    data: [di.positive_cases, di.negative_cases],
                    backgroundColor: ['#e94560','#00d4ff'],
                    borderColor: 'rgba(26,26,46,.8)',
                    borderWidth: 3, borderRadius: 8
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { position:'bottom', labels:{color:'rgba(255,255,255,.9)', padding:20, font:{size:13,weight:600}} },
                    tooltip: { backgroundColor:'rgba(26,26,46,.9)', titleColor:'#00d4ff', bodyColor:'#fff', borderColor:'rgba(0,212,255,.5)', borderWidth:1 }
                }
            }
        });

    } catch (err) {
        console.error('Error cargando datos:', err);
    }
}

loadData();
setInterval(loadData, 30000);
</script>
</body>
</html>
'''

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 DASHBOARD MÉDICO - INICIANDO")
    print("="*70)
    print("\n📊 Servidor disponible en: http://localhost:5000")
    print("🌐 Abre tu navegador y dirígete a: http://127.0.0.1:5000\n")
    print("Presiona CTRL+C para detener el servidor\n")
    print("="*70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)