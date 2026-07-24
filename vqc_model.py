"""
vqc_model.py
-------------------------------------------------
Variational Quantum Classifier (VQC) matching the methodology described
in the conference paper "Disease Risk Prediction Using Quantum
Probability Models on a Quantum Simulator: Performance Comparison with
Classical Machine Learning" (Section III-D / IV-B):

  - 13 raw Cleveland heart-disease attributes
  - StandardScaler normalization
  - PCA: 13 features -> 4 principal components (one per qubit)
  - Feature map:      RY(x_i) on each of 4 qubits   |psi(x)> = RY(x1)(x)RY(x2)(x)RY(x3)(x)RY(x4)|0000>
  - Entanglement:     full CNOT chain between adjacent qubits (0-1, 1-2, 2-3)
  - Ansatz:           trainable RX(theta_i) on each qubit
  - Readout:          expectation value of Pauli-Z on qubit 0 -> probability
                       p = (1 + <Z0>) / 2
  - Loss:             binary cross-entropy
  - Optimizer:        COBYLA, max 200 iterations
  - Classical baseline: Logistic Regression on the same preprocessed data
  - Simulator:        Qiskit Aer (exact statevector, no shot noise)

This replaces the previous placeholder ("toy single-qubit RY + threshold,
no training") which did not implement anything resembling the VQC
described in the paper.
"""

import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

N_QUBITS = 4
COBYLA_MAXITER = 200


# --------------------------------------------------------------------------
# Circuit construction
# --------------------------------------------------------------------------
def build_vqc_circuit(x, theta):
    """
    x:     length-4 array of PCA components (already scaled to safe rotation
           range) -> RY feature-map angles.
    theta: length-4 array of trainable ansatz parameters -> RX gate angles.
    Returns a QuantumCircuit implementing:
        RY feature map -> CNOT entanglement chain -> RX ansatz
    """
    qc = QuantumCircuit(N_QUBITS)

    # Feature map: RY(x_i) on each qubit
    for i in range(N_QUBITS):
        qc.ry(float(x[i]), i)

    # Entanglement: full CNOT chain between adjacent qubits
    for i in range(N_QUBITS - 1):
        qc.cx(i, i + 1)

    # Trainable ansatz: RX(theta_i) on each qubit
    for i in range(N_QUBITS):
        qc.rx(float(theta[i]), i)

    return qc


_Z0 = SparsePauliOp.from_list([("IIIZ", 1.0)])  # Pauli-Z on qubit 0 (little-endian)


def _expectation_z0(x, theta):
    qc = build_vqc_circuit(x, theta)
    state = Statevector.from_instruction(qc)
    return float(np.real(state.expectation_value(_Z0)))


def _predict_proba_single(x, theta):
    z = _expectation_z0(x, theta)
    p = (1.0 + z) / 2.0
    return min(max(p, 1e-9), 1 - 1e-9)  # clip for numerical stability in log-loss


def vqc_predict_proba(X, theta):
    return np.array([_predict_proba_single(x, theta) for x in X])


# --------------------------------------------------------------------------
# Training (cross-entropy loss, COBYLA optimizer)
# --------------------------------------------------------------------------
def _cross_entropy_loss(theta, X, y):
    p = vqc_predict_proba(X, theta)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def train_vqc(X_train, y_train, maxiter=COBYLA_MAXITER, seed=42):
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(-np.pi / 4, np.pi / 4, size=N_QUBITS)

    result = minimize(
        _cross_entropy_loss,
        theta0,
        args=(X_train, y_train),
        method="COBYLA",
        options={"maxiter": maxiter},
    )
    return result.x, result


# --------------------------------------------------------------------------
# Full pipeline: preprocessing -> classical baseline -> VQC -> comparison
# --------------------------------------------------------------------------
def run_vqc_comparison(csv_path="heart.csv", cobyla_maxiter=COBYLA_MAXITER):
    df = pd.read_csv(csv_path)
    X = df.drop("target", axis=1).values
    y = df["target"].values

    # 80/20 split as specified for the VQC experiments in the paper
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # StandardScaler normalization (paper: VQC experiments)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------- Classical baseline: Logistic Regression ----------------
    start = time.perf_counter()
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train_scaled, y_train)
    classical_training_time = time.perf_counter() - start

    start = time.perf_counter()
    classical_preds = clf.predict(X_test_scaled)
    classical_prediction_time = time.perf_counter() - start

    classical_metrics = {
        "accuracy": accuracy_score(y_test, classical_preds),
        "precision": precision_score(y_test, classical_preds),
        "recall": recall_score(y_test, classical_preds),
        "f1": f1_score(y_test, classical_preds),
    }

    # ---------------- Quantum: PCA 13 -> 4 components for the VQC ----------------
    pca = PCA(n_components=N_QUBITS, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Scale PCA components into a sensible rotation-angle range [-pi, pi]
    max_abs = np.max(np.abs(X_train_pca)) or 1.0
    X_train_angles = (X_train_pca / max_abs) * np.pi
    X_test_angles = (X_test_pca / max_abs) * np.pi

    # ---------------- VQC: 4-qubit circuit, RY feature map + CNOT + RX ansatz, COBYLA ----------------
    start = time.perf_counter()
    theta, opt_result = train_vqc(X_train_angles, y_train, maxiter=cobyla_maxiter)
    quantum_training_time = time.perf_counter() - start

    start = time.perf_counter()
    quantum_proba = vqc_predict_proba(X_test_angles, theta)
    quantum_preds = (quantum_proba > 0.5).astype(int)
    quantum_prediction_time = time.perf_counter() - start

    quantum_metrics = {
        "accuracy": accuracy_score(y_test, quantum_preds),
        "precision": precision_score(y_test, quantum_preds, zero_division=0),
        "recall": recall_score(y_test, quantum_preds, zero_division=0),
        "f1": f1_score(y_test, quantum_preds, zero_division=0),
    }

    return {
        "classical": {
            **{k: round(float(v), 4) for k, v in classical_metrics.items()},
            "training_time_s": round(float(classical_training_time), 5),
            "prediction_time_s": round(float(classical_prediction_time), 6),
        },
        "quantum_vqc": {
            **{k: round(float(v), 4) for k, v in quantum_metrics.items()},
            "training_time_s": round(float(quantum_training_time), 5),
            "prediction_time_s": round(float(quantum_prediction_time), 6),
            "cobyla_iterations": int(opt_result.nfev),
            "final_theta": [float(t) for t in theta],
        },
    }


if __name__ == "__main__":
    results = run_vqc_comparison()
    import json
    print(json.dumps(results, indent=2))
