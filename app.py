import traceback
from flask import Flask, render_template, request, jsonify
from model import predict_heart_disease
from vqc_model import run_vqc_comparison

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [
        float(data['age']),
        float(data['sex']),
        float(data['cp']),
        float(data['trestbps']),
        float(data['chol']),
        float(data['fbs']),
        float(data['restecg']),
        float(data['thalach']),
        float(data['exang']),
        float(data['oldpeak']),
        float(data['slope']),
        float(data['ca']),
        float(data['thal'])
    ]
    result = predict_heart_disease(features)
    return jsonify({'prediction': result})


@app.route('/quantum-compare', methods=['POST'])
def quantum_compare():
    """
    Runs the paper's actual comparison: classical Logistic Regression
    vs. a 4-qubit Variational Quantum Classifier (RY feature map,
    CNOT entanglement chain, RX ansatz, COBYLA optimizer), on
    PCA-reduced, StandardScaler-normalized data. This replaces the
    old single-qubit / untrained placeholder.
    """
    try:
        results = run_vqc_comparison()
        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
