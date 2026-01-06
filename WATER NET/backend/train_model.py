from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
from preprocess import load_and_preprocess

X_train, X_test, y_drink_train, y_irri_train, scaler = \
    load_and_preprocess("data/water_quality_dataset.csv")

# Drinking Water Model – Logistic Regression
drink_model = LogisticRegression()
drink_model.fit(X_train, y_drink_train)

# Irrigation Water Model – SVM
irrigation_model = SVC(kernel='rbf', probability=True)
irrigation_model.fit(X_train, y_irri_train)

joblib.dump(drink_model, "models/drinking_model.pkl")
joblib.dump(irrigation_model, "models/irrigation_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Models trained and saved successfully")
