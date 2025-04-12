import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Ścieżka do zapisanego modelu
MODEL_NAME = "saved_model.pkl"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Wczytanie modelu
path = os.path.join(BASE_DIR, MODEL_NAME)
xgb = joblib.load(path)
print("Model wczytany.")

# Przykładowe dane wejściowe (możesz podać swoje)
features = ['age', 'gender', 'restingrelectro', 'maxheartrate', 'oldpeak', 'slope']
sample_data = pd.DataFrame([[59, 1, 1, 168, 2.1, 2]], columns=features)

# Skalowanie danych (użyj tego samego skalowania co przy trenowaniu)
scaler = StandardScaler()
sample_data_scaled = scaler.fit_transform(sample_data)

# Predykcja
prediction = xgb.predict(sample_data_scaled)
print("Przewidywana wartość:", prediction)
