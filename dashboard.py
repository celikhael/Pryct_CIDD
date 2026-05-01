#!/usr/bin/env python3
"""
Dashboard Web Ultra Moderno para Modelo de Clasificación Médica
Tecnología: Flask + Chart.js + CSS3 Avanzado
"""

from flask import Flask, jsonify, render_template_string
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import os

app = Flask(__name__)

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
    
    # Calcular métricas
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
    
    # Feature importance
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
    """Retorna todas las métricas del modelo"""
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
    """Retorna las características más importantes"""
    return jsonify([
        {
            'feature': row['feature'],
            'importance': float(row['importance'])
        }
        for _, row in feature_importance.iterrows()
    ])

@app.route('/api/dataset-info')
def get_dataset_info():
    """Retorna información del dataset"""
    df_clean = df.drop('id', axis=1)
    return jsonify({
        'total_samples': len(df),
        'features': len(df_clean.columns) - 1,
        'positive_cases': int((df['diagnosis'] == 'M').sum()),
        'negative_cases': int((df['diagnosis'] == 'B').sum()),
        'test_samples': len(X_test),
        'train_samples': len(df) - len(X_test)
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
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

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
            position: relative;
        }

        /* Fondo animado */
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 50%, rgba(15, 52, 96, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(233, 69, 96, 0.1) 0%, transparent 50%);
            pointer-events: none;
            z-index: -1;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 40px 20px;
        }

        /* ===== HEADER ===== */
        .header {
            margin-bottom: 60px;
            padding-bottom: 40px;
            border-bottom: 2px solid rgba(233, 69, 96, 0.3);
            position: relative;
            overflow: hidden;
        }

        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #e94560, #00d4ff, #e94560);
            animation: slideRight 3s ease-in-out infinite;
        }

        @keyframes slideRight {
            0%, 100% { transform: translateX(-100%); }
            50% { transform: translateX(100%); }
        }

        .header-content {
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 20px;
        }

        .header-icon {
            font-size: 48px;
            animation: float 3s ease-in-out infinite;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }

        h1 {
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #00d4ff, #e94560);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -1px;
        }

        .header-subtitle {
            color: var(--color-text-secondary);
            font-size: 1.1rem;
            margin-top: 10px;
            font-weight: 400;
        }

        /* ===== MÉTRICAS PRINCIPALES ===== */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 60px;
        }

        .metric-card {
            background: linear-gradient(135deg, rgba(22, 33, 62, 0.8), rgba(15, 52, 96, 0.8));
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 16px;
            padding: 32px;
            position: relative;
            overflow: hidden;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
        }

        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.2), transparent);
            transition: left 0.5s;
        }

        .metric-card:hover {
            border-color: rgba(0, 212, 255, 0.5);
            transform: translateY(-8px);
            box-shadow: 0 20px 60px rgba(0, 212, 255, 0.15);
        }

        .metric-card:hover::before {
            left: 100%;
        }

        .metric-label {
            font-size: 0.95rem;
            color: var(--color-text-secondary);
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 12px;
            font-weight: 600;
            font-family: var(--font-mono);
        }

        .metric-value {
            font-size: 3.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #00d4ff, #00d4ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-family: var(--font-mono);
        }

        .metric-card.success .metric-value {
            background: linear-gradient(135deg, #00ff88, #00d4ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .metric-card.warning .metric-value {
            background: linear-gradient(135deg, #ffd700, #ffaa00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .metric-card.danger .metric-value {
            background: linear-gradient(135deg, #e94560, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* ===== GRID DE CONTENIDO ===== */
        .content-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-bottom: 60px;
        }

        @media (max-width: 1200px) {
            .content-grid {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: linear-gradient(135deg, rgba(22, 33, 62, 0.6), rgba(15, 52, 96, 0.6));
            border: 1px solid rgba(233, 69, 96, 0.2);
            border-radius: 16px;
            padding: 32px;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(10px);
        }

        .card::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(0, 212, 255, 0.1) 0%, transparent 70%);
            pointer-events: none;
        }

        .card-title {
            font-size: 1.6rem;
            font-weight: 700;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 12px;
            border-bottom: 2px solid rgba(233, 69, 96, 0.3);
            padding-bottom: 16px;
            position: relative;
            z-index: 1;
        }

        .card-title-icon {
            font-size: 1.8rem;
        }

        /* ===== MATRIZ DE CONFUSIÓN ===== */
        .confusion-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            position: relative;
            z-index: 1;
        }

        .confusion-cell {
            background: rgba(255, 255, 255, 0.05);
            border: 2px solid rgba(0, 212, 255, 0.3);
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .confusion-cell::before {
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at center, rgba(0, 212, 255, 0.1), transparent);
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .confusion-cell:hover {
            border-color: rgba(0, 212, 255, 0.7);
            transform: scale(1.05);
        }

        .confusion-cell:hover::before {
            opacity: 1;
        }

        .confusion-label {
            font-size: 0.85rem;
            color: var(--color-text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
            font-weight: 600;
            font-family: var(--font-mono);
        }

        .confusion-value {
            font-size: 2.8rem;
            font-weight: 700;
            font-family: var(--font-mono);
            color: #00d4ff;
        }

        .confusion-cell.positive .confusion-value {
            color: #00ff88;
        }

        .confusion-cell.negative .confusion-value {
            color: #ffd700;
        }

        .confusion-cell.error .confusion-value {
            color: #e94560;
        }

        /* ===== CARACTERÍSTICAS ===== */
        .features-list {
            position: relative;
            z-index: 1;
        }

        .feature-item {
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid rgba(233, 69, 96, 0.2);
            animation: fadeInUp 0.6s ease-out backwards;
        }

        .feature-item:last-child {
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .feature-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .feature-name {
            font-weight: 600;
            color: var(--color-text);
            font-size: 0.95rem;
        }

        .feature-percentage {
            background: rgba(0, 212, 255, 0.2);
            color: #00d4ff;
            padding: 4px 12px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            font-family: var(--font-mono);
        }

        .feature-bar {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
            position: relative;
        }

        .feature-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #e94560);
            border-radius: 4px;
            animation: expandWidth 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) backwards;
        }

        @keyframes expandWidth {
            from { width: 0; }
        }

        /* ===== GRÁFICOS ===== */
        .chart-container {
            position: relative;
            height: 350px;
            margin-bottom: 20px;
            z-index: 1;
        }

        canvas {
            filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.1));
        }

        /* ===== ESTADÍSTICAS ===== */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
            position: relative;
            z-index: 1;
            margin-top: 24px;
            padding-top: 24px;
            border-top: 1px solid rgba(233, 69, 96, 0.3);
        }

        .stat-box {
            background: rgba(0, 212, 255, 0.05);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            transition: all 0.3s ease;
        }

        .stat-box:hover {
            border-color: rgba(0, 212, 255, 0.5);
            background: rgba(0, 212, 255, 0.1);
        }

        .stat-label {
            font-size: 0.8rem;
            color: var(--color-text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
            font-family: var(--font-mono);
        }

        .stat-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #00d4ff;
            font-family: var(--font-mono);
        }

        /* ===== FOOTER ===== */
        .footer {
            text-align: center;
            padding: 40px 20px;
            color: var(--color-text-secondary);
            font-size: 0.9rem;
            border-top: 1px solid rgba(233, 69, 96, 0.2);
            margin-top: 60px;
        }

        .footer p {
            margin: 8px 0;
        }

        /* ===== RESPONSIVE ===== */
        @media (max-width: 768px) {
            h1 {
                font-size: 2.2rem;
            }

            .metrics-grid {
                grid-template-columns: 1fr;
            }

            .metric-card {
                padding: 24px;
            }

            .metric-value {
                font-size: 2.4rem;
            }

            .confusion-grid {
                grid-template-columns: 1fr 1fr;
            }

            .stats-grid {
                grid-template-columns: 1fr;
            }
        }

        /* ===== SCROLLBAR ===== */
        ::-webkit-scrollbar {
            width: 10px;
        }

        ::-webkit-scrollbar-track {
            background: var(--color-primary);
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #00d4ff, #e94560);
            border-radius: 5px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #00ffff, #ff1493);
        }

        /* Loading */
        .loading {
            opacity: 0.5;
            pointer-events: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="header-content">
                <div class="header-icon">🏥</div>
                <div>
                    <h1>DIAGNÓSTICO INTELIGENTE</h1>
                    <p class="header-subtitle">Sistema de Clasificación Médica Avanzado</p>
                </div>
            </div>
        </div>

        <!-- Métricas Principales -->
        <div class="metrics-grid" id="metricsContainer">
            <div class="metric-card success">
                <div class="metric-label">Precisión General</div>
                <div class="metric-value" id="accuracy">0%</div>
            </div>
            <div class="metric-card success">
                <div class="metric-label">Precisión Positivos</div>
                <div class="metric-value" id="precision">0%</div>
            </div>
            <div class="metric-card success">
                <div class="metric-label">Sensibilidad</div>
                <div class="metric-value" id="recall">0%</div>
            </div>
            <div class="metric-card warning">
                <div class="metric-label">F1 Score</div>
                <div class="metric-value" id="f1">0%</div>
            </div>
            <div class="metric-card success">
                <div class="metric-label">AUC-ROC</div>
                <div class="metric-value" id="auc">0%</div>
            </div>
        </div>

        <!-- Contenido Principal -->
        <div class="content-grid">
            <!-- Matriz de Confusión -->
            <div class="card">
                <div class="card-title">
                    <span class="card-title-icon">📊</span>
                    Matriz de Confusión
                </div>
                <div class="confusion-grid">
                    <div class="confusion-cell negative">
                        <div class="confusion-label">Verdaderos Negativos</div>
                        <div class="confusion-value" id="tn">0</div>
                    </div>
                    <div class="confusion-cell error">
                        <div class="confusion-label">Falsos Positivos</div>
                        <div class="confusion-value" id="fp">0</div>
                    </div>
                    <div class="confusion-cell error">
                        <div class="confusion-label">Falsos Negativos</div>
                        <div class="confusion-value" id="fn">0</div>
                    </div>
                    <div class="confusion-cell positive">
                        <div class="confusion-label">Verdaderos Positivos</div>
                        <div class="confusion-value" id="tp">0</div>
                    </div>
                </div>
            </div>

            <!-- Top Características -->
            <div class="card">
                <div class="card-title">
                    <span class="card-title-icon">⭐</span>
                    Características Principales
                </div>
                <div class="features-list" id="featuresContainer">
                    <div style="text-align: center; color: var(--color-text-secondary);">Cargando...</div>
                </div>
            </div>
        </div>

        <!-- Gráficos -->
        <div class="content-grid">
            <div class="card">
                <div class="card-title">
                    <span class="card-title-icon">📈</span>
                    Importancia de Características
                </div>
                <div class="chart-container">
                    <canvas id="featureChart"></canvas>
                </div>
            </div>

            <div class="card">
                <div class="card-title">
                    <span class="card-title-icon">📉</span>
                    Distribución de Dataset
                </div>
                <div class="chart-container">
                    <canvas id="datasetChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Estadísticas Completas -->
        <div class="card">
            <div class="card-title">
                <span class="card-title-icon">📚</span>
                Información del Dataset
            </div>
            <div class="stats-grid" id="statsContainer">
                <div class="stat-box">
                    <div class="stat-label">Muestras Totales</div>
                    <div class="stat-value" id="totalSamples">0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Características</div>
                    <div class="stat-value" id="numFeatures">0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Casos Positivos</div>
                    <div class="stat-value" id="positiveCases">0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Casos Negativos</div>
                    <div class="stat-value" id="negativeCases">0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Muestras Entrenamiento</div>
                    <div class="stat-value" id="trainSamples">0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Muestras Prueba</div>
                    <div class="stat-value" id="testSamples">0</div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>🤖 Powered by Random Forest | Flask + Chart.js | Sistema de Diagnóstico Inteligente</p>
            <p style="margin-top: 12px; font-size: 0.8rem;">© 2024 Machine Learning Dashboard</p>
        </div>
    </div>

    <script>
        // Variables globales para gráficos
        let featureChart = null;
        let datasetChart = null;

        // Cargar datos del backend
        async function loadData() {
            try {
                // Cargar métricas
                const metricsRes = await fetch('/api/metrics');
                const metrics = await metricsRes.json();

                // Actualizar métricas
                document.getElementById('accuracy').textContent = (metrics.accuracy * 100).toFixed(1) + '%';
                document.getElementById('precision').textContent = (metrics.precision * 100).toFixed(1) + '%';
                document.getElementById('recall').textContent = (metrics.recall * 100).toFixed(1) + '%';
                document.getElementById('f1').textContent = (metrics.f1 * 100).toFixed(1) + '%';
                document.getElementById('auc').textContent = (metrics.auc * 100).toFixed(1) + '%';

                // Matriz de confusión
                document.getElementById('tn').textContent = metrics.confusion_matrix.tn;
                document.getElementById('fp').textContent = metrics.confusion_matrix.fp;
                document.getElementById('fn').textContent = metrics.confusion_matrix.fn;
                document.getElementById('tp').textContent = metrics.confusion_matrix.tp;

                // Cargar características
                const featuresRes = await fetch('/api/features');
                const features = await featuresRes.json();
                
                const featuresHtml = features.map((f, i) => `
                    <div class="feature-item" style="animation-delay: ${i * 50}ms">
                        <div class="feature-header">
                            <span class="feature-name">${f.feature}</span>
                            <span class="feature-percentage">${(f.importance * 100).toFixed(2)}%</span>
                        </div>
                        <div class="feature-bar">
                            <div class="feature-bar-fill" style="width: ${f.importance * 100}%; animation-delay: ${i * 50}ms"></div>
                        </div>
                    </div>
                `).join('');
                document.getElementById('featuresContainer').innerHTML = featuresHtml;

                // Cargar información del dataset
                const datasetRes = await fetch('/api/dataset-info');
                const datasetInfo = await datasetRes.json();

                document.getElementById('totalSamples').textContent = datasetInfo.total_samples;
                document.getElementById('numFeatures').textContent = datasetInfo.features;
                document.getElementById('positiveCases').textContent = datasetInfo.positive_cases;
                document.getElementById('negativeCases').textContent = datasetInfo.negative_cases;
                document.getElementById('trainSamples').textContent = datasetInfo.train_samples;
                document.getElementById('testSamples').textContent = datasetInfo.test_samples;

                // Gráfico de características
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
                                backgroundColor: features.map((_, i) => {
                                    const colors = ['#00d4ff', '#00ff88', '#ffd700', '#e94560'];
                                    return colors[i % colors.length];
                                }),
                                borderColor: 'rgba(0, 212, 255, 0.3)',
                                borderWidth: 2,
                                borderRadius: 8,
                                hoverBackgroundColor: '#e94560'
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            indexAxis: 'y',
                            plugins: {
                                legend: { display: false },
                                tooltip: {
                                    backgroundColor: 'rgba(26, 26, 46, 0.9)',
                                    titleColor: '#00d4ff',
                                    bodyColor: '#ffffff',
                                    borderColor: 'rgba(0, 212, 255, 0.5)',
                                    borderWidth: 1,
                                    padding: 12
                                }
                            },
                            scales: {
                                x: {
                                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                                    ticks: { color: 'rgba(255, 255, 255, 0.7)' },
                                    max: 20
                                },
                                y: {
                                    grid: { display: false },
                                    ticks: { color: 'rgba(255, 255, 255, 0.7)' }
                                }
                            }
                        }
                    });
                }

                // Gráfico de distribución
                const ctx2 = document.getElementById('datasetChart').getContext('2d');
                if (datasetChart) datasetChart.destroy();
                
                datasetChart = new Chart(ctx2, {
                    type: 'doughnut',
                    data: {
                        labels: ['Positivos (M)', 'Negativos (B)'],
                        datasets: [{
                            data: [datasetInfo.positive_cases, datasetInfo.negative_cases],
                            backgroundColor: ['#e94560', '#00d4ff'],
                            borderColor: 'rgba(26, 26, 46, 0.8)',
                            borderWidth: 3,
                            borderRadius: 8
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: 'bottom',
                                labels: {
                                    color: 'rgba(255, 255, 255, 0.9)',
                                    padding: 20,
                                    font: { size: 13, weight: 600 }
                                }
                            },
                            tooltip: {
                                backgroundColor: 'rgba(26, 26, 46, 0.9)',
                                titleColor: '#00d4ff',
                                bodyColor: '#ffffff',
                                borderColor: 'rgba(0, 212, 255, 0.5)',
                                borderWidth: 1
                            }
                        }
                    }
                });

            } catch (error) {
                console.error('Error cargando datos:', error);
            }
        }

        // Cargar datos al iniciar
        loadData();

        // Actualizar cada 30 segundos
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
