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

# Globalne obliczenia cech
resting_df = df[df["Activity Level Numeric"] < 2]
global_restingrelectro = resting_df["Heart Rate (BPM)"].mean() if not resting_df.empty else np.nan
print("\nGlobalny średni puls w stanie spoczynku:", global_restingrelectro)
age = 53  # przykładowy wiek
maxheartrate = 220 - age  # standardowy maksymalny puls (220 - wiek)
print("Standardowy maksymalny puls (HRmax):", maxheartrate)
global_oldpeak = maxheartrate - global_restingrelectro if not np.isnan(global_restingrelectro) else np.nan
print("Globalny oldpeak:", global_oldpeak)
lr_data = df[['Step Count', 'Heart Rate (BPM)']].copy()
lr_data['Step Count'] = pd.to_numeric(lr_data['Step Count'], errors='coerce')
lr_data['Heart Rate (BPM)'] = pd.to_numeric(lr_data['Heart Rate (BPM)'], errors='coerce')
lr_data = lr_data.dropna()
X = lr_data["Step Count"].values.reshape(-1, 1)
y = lr_data["Heart Rate (BPM)"].values
lr_model = LinearRegression()
lr_model.fit(X, y)
slope = lr_model.coef_[0]
print("Globalny slope (współczynnik):", slope)
gender = 1  # przykładowa płeć

url = "http://172.16.16.13:8080"
headers = {"Content-Type": "application/json"}

# Losujemy rekordy i wysyłamy dane, dopóki serwer nie zwróci "Choroba serca"
while True:
    random_index = df.sample(1).index[0]  # losowanie indeksu
    record = df.loc[random_index]          # pobranie rekordu
    user_id_value = record["User ID"] if "User ID" in df.columns else None  # pobranie User ID
    # Jeśli wylosowany rekord jest w stanie spoczynku, użyj jego HR, w przeciwnym razie globalnego HR
    local_restingrelectro = record["Heart Rate (BPM)"] if (pd.notna(record["Activity Level Numeric"]) and record["Activity Level Numeric"] < 2) else global_restingrelectro
    local_oldpeak = maxheartrate - local_restingrelectro if not np.isnan(local_restingrelectro) else np.nan
    data_payload = {
        "model": "heart_model",
        "data": {
            "user_id": user_id_value,
            "age": age,
            "gender": gender,
            "restingrelectro": round(local_restingrelectro, 2) if not np.isnan(local_restingrelectro) else None,
            "maxheartrate": maxheartrate,
            "oldpeak": round(local_oldpeak, 2) if not np.isnan(local_oldpeak) else None,
            "slope": round(slope, 2)
        }
    }
    json_data = json.dumps(data_payload)
    print("\nWybrano rekord o User ID:", user_id_value)
    print("Przygotowany ładunek JSON:")
    print(json_data)
    try:
        response = requests.post(url, data=json_data, headers=headers)
        print("Status odpowiedzi:", response.status_code)
        print("Treść odpowiedzi:", response.text)
        try:
            result = response.json()
            if result.get("prediction") == "Choroba serca":
                print("Serwer zwrócił 'Choroba serca'. Zatrzymanie pętli.")
                break
        except Exception as e:
            print("Błąd parsowania JSON odpowiedzi:", e)
    except Exception as e:
        print("Wystąpił błąd podczas wysyłania żądania:", str(e))
