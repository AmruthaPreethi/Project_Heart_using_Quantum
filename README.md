# Heart Disease Prediction Web Application (with Quantum Comparison)

A Machine Learning–based web application that predicts possible heart disease from
patient data, with an added panel that benchmarks a classical model against a
simulated quantum classifier.

---

## Features

- Symptom/clinical-data based heart disease prediction (Random Forest)
- Real-time prediction through a Flask API
- Classical vs. Quantum comparison panel:
  - Trains a classical Logistic Regression model
  - Runs a simulated single-qubit "quantum" classifier (Qiskit Aer)
  - Compares prediction time and accuracy
  - Renders a bar chart of the timing comparison
- Clean, minimal web interface

---

## Tech Stack

- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn
- **Quantum Simulation:** Qiskit, Qiskit Aer
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Frontend:** HTML5, CSS3, vanilla JS

---

## Project Structure

```
Project_Heart_Full/
│
├── static/
│   ├── style.css
│   └── images/
│       └── comparison.png        (generated at runtime)
│
├── templates/
│   └── index.html
│
├── heart.csv
├── model.py                      # classical prediction model
├── quantum_model.py              # quantum vs classical comparison logic
├── app.py                        # Flask routes
├── requirements.txt
└── README.md
```

---

## Running Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000/` in your browser.

---

## Routes

| Route              | Method | Description                                      |
|---------------------|--------|---------------------------------------------------|
| `/`                 | GET    | Serves the main page                              |
| `/predict`          | POST   | Returns a heart disease prediction from features   |
| `/quantum-compare`  | POST   | Runs the classical vs quantum benchmark and returns timing/accuracy + chart |

---

## Disclaimer

This application is for educational purposes only and should not be used as a
substitute for professional medical advice. The quantum classifier is a
simplified demonstration (single qubit, single feature) and is not intended
to outperform the classical model — it's included to illustrate the
classical/quantum pipeline side by side.

---

## Author

DORADLA AMRUTHA PREETHI
B.Tech – Artificial Intelligence & Machine Learning
