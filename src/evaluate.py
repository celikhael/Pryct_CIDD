import joblib

from sklearn.metrics import accuracy_score


def evaluate_model():
    model = joblib.load("model/modelo.pkl")
    X_test = joblib.load("model/X_test.pkl")
    y_test = joblib.load("model/y_test.pkl")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100

    return accuracy
