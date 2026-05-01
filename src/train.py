import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model():
    df = pd.read_csv("data/data.csv")

    # Eliminar columna que no aporta al modelo
    df = df.drop("id", axis=1)

    # Convertir columna diagnosis a valores numéricos
    df["diagnosis"] = (df["diagnosis"] == "M").astype(int)

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

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

    print("Modelo Random Forest entrenado y guardado correctamente")