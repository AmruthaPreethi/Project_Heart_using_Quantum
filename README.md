# Heart Disease Prediction with a Quantum Comparison

This is a small web app I built to predict heart disease risk from patient data. It started
as a basic Flask + scikit-learn project, and I later added a second panel that compares the
classical model against a simulated quantum classifier, mostly out of curiosity about how
the two stack up in terms of speed and accuracy.

## What it does

The main page has a form where you enter clinical data (age, blood pressure, cholesterol,
etc.) and it predicts whether the patient is likely to have heart disease, using a Randomn
Forest model trained on the UCI heart disease dataset.

Below that there's a "Classical vs Quantum Comparison" section. Clicking the button trains
a Logistic Regression model and, alongside it, runs a very simple single-qubit quantum
circuit (simulated using Qiskit Aer) on the same test data. It shows how long each one takes
to predict and how accurate each one is, plus a bar chart comparing the prediction times.

Worth being upfront about this: the "quantum" model here is a toy example — one qubit, one
feature, a basic probability threshold. It's not meant to outperform the classical model,
and it doesn't. It's there to actually show a working classical/quantum comparison rather
than just talk about one.

## Tech used

- Python / Flask for the backend
- scikit-learn for the classical model
- Qiskit + Qiskit Aer for the quantum simulation
- Pandas / NumPy for data handling
- Matplotlib for the comparison chart
- Plain HTML/CSS/JS for the frontend, no framework

## Project structure
Project_Heart_Full/
├── static/
│ ├── style.css
│ └── images/ # comparison chart gets saved here when you run it
├── templates/
│ └── index.html
├── heart.csv
├── model.py # classical prediction model
├── quantum_model.py # classical vs quantum comparison logic
├── app.py # Flask app and routes
├── requirements.txt
└── README.md

## Running it

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Routes

- `GET /` – loads the main page
- `POST /predict` – takes patient data and returns a prediction
- `POST /quantum-compare` – runs the classical vs quantum benchmark and returns the results

## Disclaimer

This is a student/educational project, not a medical tool. Don't use it to make actual
health decisions. The quantum part especially is a simplified demo, not a real diagnostic
model.

## Author

Doradla Amrutha Preethi
B.Tech – AI & ML
