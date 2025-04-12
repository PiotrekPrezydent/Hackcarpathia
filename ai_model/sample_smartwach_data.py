import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import json
import requests
from sklearn.linear_model import LinearRegression

# Wczytanie datasetu
file_path = "unclean_smartwatch_health_data.csv"
df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "mohammedarfathr/smartwatch-health-data-uncleaned", file_path)
print("Kolumny w zbiorze danych:")
print(df.columns.tolist())
print("\nPierwsze 5 rekordów (pierwsze 5 kolumn):")
print(df.iloc[:, :5].head())
print("\nPierwsze 5 rekordów całego DataFrame:")
print(df.head())

# Normalizacja kolumny 'Activity Level'
activity_map = {"sedentary": 0, "low": 1, "active": 2, "actve": 2, "high active": 3, "highly active": 3}
df["Activity Level Numeric"] = df["Activity Level"].str.lower().str.replace("_", " ", regex=False).str.strip().map(activity_map)
print("\nPodgląd 'Activity Level' vs. 'Activity Level Numeric':")
print(df[["Activity Level", "Activity Level Numeric"]].head())

# Obliczenia cech
resting_df = df[df["Activity Level Numeric"] < 2]  # wyliczenie spoczynkowego pulsu
restingrelectro = resting_df["Heart Rate (BPM)"].mean() if not resting_df.empty else np.nan
print("\nŚredni puls w stanie spoczynku:", restingrelectro)
age = 53  # przykładowy wiek
maxheartrate = 220 - age  # standardowy maksymalny puls (220 - wiek)
print("Standardowy maksymalny puls (HRmax):", maxheartrate)
oldpeak = maxheartrate - restingrelectro if not np.isnan(restingrelectro) else np.nan  # różnica między HRmax a puls spoczynkowym
print("Oldpeak:", oldpeak)
lr_data = df[['Step Count', 'Heart Rate (BPM)']].copy()  # przygotowanie danych do regresji
lr_data['Step Count'] = pd.to_numeric(lr_data['Step Count'], errors='coerce')
lr_data['Heart Rate (BPM)'] = pd.to_numeric(lr_data['Heart Rate (BPM)'], errors='coerce')
lr_data = lr_data.dropna()
X = lr_data["Step Count"].values.reshape(-1, 1)
y = lr_data["Heart Rate (BPM)"].values
lr_model = LinearRegression()
lr_model.fit(X, y)
slope = lr_model.coef_[0]
print("Slope (współczynnik):", slope)

# Pobranie losowego rekordu z DataFrame
random_index = df.sample(1).index[0]
print("Wybrano losowy indeks:", random_index)
user_id_value = df.loc[random_index, "User ID"] if "User ID" in df.columns else None
print("User ID z losowych danych:", user_id_value)
print("\nDane źródłowe z wybranego rekordu:")
print(df.loc[random_index])

# Przygotowanie danych do wysyłki
gender = 1  # przykładowa płeć (1 = mężczyzna, 0 = kobieta)
data_payload = {
    "model": "heart_model",
    "data": {
        "user_id": user_id_value,
        "age": age,
        "gender": gender,
        "restingrelectro": round(restingrelectro, 2) if not np.isnan(restingrelectro) else None,
        "maxheartrate": maxheartrate,
        "oldpeak": round(oldpeak, 2) if not np.isnan(oldpeak) else None,
        "slope": round(slope, 2)
    }
}
json_data = json.dumps(data_payload)
print("\nPrzygotowany ładunek JSON do wysłania:")
print(json_data)

# Wysłanie żądania POST
url = "http://172.16.16.13:8080"
headers = {"Content-Type": "application/json"}
try:
    response = requests.post(url, data=json_data, headers=headers)
    print("\nStatus odpowiedzi:", response.status_code)
    print("Treść odpowiedzi:", response.text)
except Exception as e:
    print("Wystąpił błąd podczas wysyłania żądania:", str(e))
