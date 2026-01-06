import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    X = df.drop(['drinking', 'irrigation'], axis=1)
    y_drink = df['drinking']
    y_irri = df['irrigation']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_drink_train, y_drink_test, y_irri_train, y_irri_test = \
        train_test_split(X_scaled, y_drink, y_irri, test_size=0.2, random_state=42)

    return X_train, X_test, y_drink_train, y_irri_train, scaler
