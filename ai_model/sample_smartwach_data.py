import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import json
import requests
from sklearn.linear_model import LinearRegression

# Wczytanie datasetu – plik CSV zawiera kolumny:
# User ID, Heart Rate (BPM), Blood Oxygen Level (%), Step Count,
# Sleep Duration (hours), Activity Level, Stress Level, Activity Level Numeric
file_path = "unclean_smartwatch_health_data.csv"
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "mohammedarfathr/smartwatch-health-data-uncleaned",
    file_path
)

print("Kolumny w zbiorze danych:")
print(df.columns.tolist())
print("\nPierwsze 5 rekordów:")
print(df.head())

# Normalizacja kolumny 'Activity Level'
activity_map = {
    "sedentary": 0,
    "low": 1,
    "active": 2,
    "actve": 2,  # uwzględniamy literówkę
    "high active": 3,
    "highly active": 3
}
df["Activity Level Numeric"] = (
    df["Activity Level"]
    .str.lower()
    .str.replace("_", " ", regex=False)
    .str.strip()
    .map(activity_map)
)
print("\nPodgląd kolumn 'Activity Level' oraz 'Activity Level Numeric':")
print(df[["Activity Level", "Activity Level Numeric"]].head())

# Globalne obliczenie puls w spoczynku (restingrelectro)
# Używamy średniej wartości "Heart Rate (BPM)" dla rekordów, gdzie Activity Level Numeric < 2
resting_df = df[df["Activity Level Numeric"] < 2]
global_resting_relectro = (
    resting_df["Heart Rate (BPM)"].mean() if not resting_df.empty else np.nan
)
print("\nGlobalny puls w spoczynku (restingrelectro):", global_resting_relectro)

# Ustalamy stałą wartość maksymalnego pulsu – zgodnie z danymi treningowymi
# Dodajemy losowy szum w zakresie -10 do 10
maxheartrate = int(round(177.6 + np.random.uniform(-10, 10)))
print("Maksymalny puls (maxheartrate) z dodanym szumem jako int:", maxheartrate)

# Obliczenie średnich wartości dla brakujących danych
mean_restingrelectro = int(round(df["Heart Rate (BPM)"].mean())) if not df[
    "Heart Rate (BPM)"].empty else 70  # Domyślne 70
mean_oldpeak = int(round((maxheartrate - mean_restingrelectro) / 2))  # Przyjmijmy połowę max-oldpeak jako zastępstwo

# Globalny oldpeak obliczamy jako różnicę między maxheartrate a globalnym puls (jeśli puls < maxheartrate)
# Dodajemy losowy szum w zakresie -3 do 3
global_oldpeak = (
    int(round((maxheartrate - global_resting_relectro + np.random.uniform(-3, 3))))
    if global_resting_relectro < maxheartrate else mean_oldpeak
)
print("Globalny oldpeak z dodanym szumem jako int:", global_oldpeak)

# Obliczenie współczynnika nachylenia (slope) przy użyciu regresji liniowej między 'Step Count' a 'Heart Rate (BPM)'
lr_data = df[['Step Count', 'Heart Rate (BPM)']].copy()
lr_data['Step Count'] = pd.to_numeric(lr_data['Step Count'], errors='coerce')
lr_data['Heart Rate (BPM)'] = pd.to_numeric(lr_data['Heart Rate (BPM)'], errors='coerce')
lr_data = lr_data.dropna()

X = lr_data["Step Count"].values.reshape(-1, 1)
y = lr_data["Heart Rate (BPM)"].values
lr_model = LinearRegression()
lr_model.fit(X, y)
computed_slope = lr_model.coef_[0]

# Przypisanie wartości slope zgodnie z kryteriami
if computed_slope > 0:
    slope_value = 1  # upsloping
elif computed_slope == 0:
    slope_value = 2  # flat
else:
    slope_value = 3  # downsloping

print("Obliczony współczynnik nachylenia (slope):", computed_slope)
print("Przypisana wartość slope:", slope_value)

# Ustalanie adresu URL serwera
url = "http://172.16.16.13:8080"
headers = {"Content-Type": "application/json"}

# Zmienne do zliczania chorych i zdrowych
chorych = 0
zdrowych = 0


# Losowanie pojedynczego rekordu
record = df.sample(1).iloc[0]

# Losowe generowanie danych:
# age: losowa wartość z przedziału 50 do 80
age_val = np.random.randint(50, 81)
# gender: losowa wartość 0 lub 1
gender_val = int(np.random.choice([0, 1]))

# Wyznaczamy lokalny puls spoczynkowy ("restingrelectro")
if pd.notna(record["Activity Level Numeric"]) and record["Activity Level Numeric"] < 2:
    local_resting = record["Heart Rate (BPM)"]
else:
    local_resting = global_resting_relectro

# Obliczenie lokalnego oldpeak – różnica między maxheartrate a lokalnym tętnem spoczynkowym
if pd.notna(local_resting) and local_resting < maxheartrate:
    local_oldpeak = maxheartrate - local_resting + np.random.uniform(-3, 3)  # Dodajemy losowy szum
else:
    local_oldpeak = mean_oldpeak  # Średnia zastępcza dla oldpeak w przypadku nan

# Przygotowanie ładunku JSON – wszystkie wartości są castowane do intów:
data_payload = {
    "model": "heart_model",
    "data": {
        "age": int(age_val),
        "gender": int(gender_val),
        "restingrelectro": int(round(local_resting)) if pd.notna(local_resting) else mean_restingrelectro,
        "maxheartrate": int(round(maxheartrate)),
        "oldpeak": int(round(local_oldpeak)) if pd.notna(local_oldpeak) else mean_oldpeak,
        "slope": int(slope_value)
    }
}

json_data = json.dumps(data_payload)

print("\nWybrano rekord z wygenerowanymi danymi:")
print(json_data)

try:
    response = requests.post(url, data=json_data, headers=headers)
    print("Status odpowiedzi:", response.status_code)
    print("Treść odpowiedzi:", response.text)

    try:
        result = response.json()
        prediction = result.get("prediction", "")
        if prediction == "Choroba serca":
            chorych += 1
            print("Dodano do liczby chorych: 1")
        elif prediction == "Brak choroby serca":
            zdrowych += 1
            print("Dodano do liczby zdrowych: 1")
    except Exception as e:
        print("Błąd parsowania JSON odpowiedzi:", e)
except Exception as e:
    print("Błąd podczas wysyłania żądania:", e)


# Wyświetlenie podsumowania
print(f"\nLiczba zdrowych: {zdrowych}")
print(f"Liczba potencjalnie chorych: {chorych}")
