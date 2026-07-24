import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

model = None
scaler = None


def train_model():
    global model, scaler
    df = pd.read_csv('heart.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_scaled, y)


def predict_heart_disease(features):
    global model, scaler
    if model is None or scaler is None:
        train_model()
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    return int(prediction)
