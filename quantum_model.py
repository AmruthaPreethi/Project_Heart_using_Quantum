"""
quantum_model.py
-------------------------------------------------
Runs a classical Logistic Regression model against a toy
single-qubit "quantum" classifier (simulated with Qiskit Aer)
on heart.csv, and returns timing/accuracy results plus a
saved bar chart image for the frontend to display.

Note: the quantum classifier here is a simple demonstration
(1 feature -> 1 qubit rotation -> threshold), not a production
quantum ML model, so lower accuracy vs. the classical model is
expected. It's meant to show the classical-vs-quantum pipeline
and compare timing side by side.
"""

import os
import time
import matplotlib
matplotlib.use("Agg")  # no display server available on the backend
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

CHART_PATH = os.path.join("static", "images", "comparison.png")


def _quantum_predict(sample, simulator, shots=1000):
    qc = QuantumCircuit(1, 1)
    angle = float(sample[0])
    qc.ry(angle, 0)
    qc.measure(0, 0)

    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts()
    prob_1 = counts.get('1', 0) / shots
    return 1 if prob_1 > 0.5 else 0


def run_comparison(csv_path="heart.csv"):
    data = pd.read_csv(csv_path)
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- Classical ---
    start = time.time()
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train, y_train)
    classical_training_time = time.time() - start

    start = time.time()
    classical_preds = clf.predict(X_test)
    classical_prediction_time = time.time() - start
    classical_accuracy = accuracy_score(y_test, classical_preds)

    # --- Quantum (simulated) ---
    simulator = AerSimulator()
    start = time.time()
    quantum_preds = [_quantum_predict(sample, simulator) for sample in X_test]
    quantum_prediction_time = time.time() - start
    quantum_accuracy = accuracy_score(y_test, quantum_preds)

    # --- Chart ---
    os.makedirs(os.path.dirname(CHART_PATH), exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.bar(
        ["Classical", "Quantum"],
        [classical_prediction_time, quantum_prediction_time],
        color=["#4C72B0", "#DD8452"],
    )
    plt.xlabel("Model")
    plt.ylabel("Prediction Time (seconds)")
    plt.title("Prediction Time Comparison")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    plt.close()

    return {
        "classical_training_time": round(classical_training_time, 5),
        "classical_prediction_time": round(classical_prediction_time, 5),
        "classical_accuracy": round(classical_accuracy, 4),
        "quantum_prediction_time": round(quantum_prediction_time, 5),
        "quantum_accuracy": round(quantum_accuracy, 4),
        "chart_url": "/" + CHART_PATH.replace(os.sep, "/"),
    }
