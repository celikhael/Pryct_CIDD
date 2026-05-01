import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def _load_test_data():
    model = joblib.load("model/modelo.pkl")
    X_test = joblib.load("model/X_test.pkl")
    y_test = joblib.load("model/y_test.pkl")
    return model, X_test, y_test


def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_
    )
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusion")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(model, X_test, y_test):
    if not hasattr(model, "predict_proba"):
        raise ValueError("El modelo no tiene metodo predict_proba.")

    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.title("Curva ROC")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, X_test):
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Este modelo no provee importancias de caracteristicas.")

    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    labels = X_test.columns[indices]
    top_n = min(10, len(labels))

    plt.figure(figsize=(10, 6))
    plt.barh(labels[:top_n][::-1], importances[indices][:top_n][::-1], color="#ff7f0e")
    plt.title("Importancia de las caracteristicas")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.show()


def show_model_graphs():
    model, X_test, y_test = _load_test_data()
    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)
    plot_feature_importance(model, X_test)
