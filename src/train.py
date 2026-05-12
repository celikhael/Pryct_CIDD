import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_model():
    df = pd.read_csv("data/data.csv")

    # Eliminar columna id (no aporta al modelo)
    df = df.drop("id", axis=1)

    # Eliminar columna fantasma "Unnamed: 32" que genera el CSV de Kaggle
    # al tener una coma extra al final de cada fila
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        print(f"Columnas vacías eliminadas: {unnamed_cols}")

    # Convertir diagnosis a valor numérico (M=1, B=0)
    df["diagnosis"] = (df["diagnosis"] == "M").astype(int)

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    print(f"Features usadas para entrenar ({len(X.columns)}): {list(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "model/modelo.pkl")
    joblib.dump(X_test, "model/X_test.pkl")
    joblib.dump(y_test, "model/y_test.pkl")

    print("Modelo Random Forest entrenado y guardado correctamente.")
