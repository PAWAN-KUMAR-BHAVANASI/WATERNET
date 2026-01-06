import joblib
import numpy as np

drink_model = joblib.load("models/drinking_model.pkl")
irrigation_model = joblib.load("models/irrigation_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict_water_quality(values):
    values = np.array(values).reshape(1, -1)
    values = scaler.transform(values)

    drinking_result = drink_model.predict(values)[0]
    irrigation_result = irrigation_model.predict(values)[0]

    return drinking_result, irrigation_result
