import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def select(X, y, features):
    # Selekcja cech
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    scores = selector.scores_

    # Wybór najlepszych cech na podstawie wyników
    selected_features = selector.get_support(indices=True)
    print("Wybrane cechy:", [features[i] for i in selected_features])

    return selected_features  # Zwraca indeksy wybranych cech

def AiModel():
    path = os.path.join(BASE_DIR, f"datasets/Cardiovascular_Disease_Dataset.csv")
    print(path)

    # Wczytywanie i filtracja danych
    df = pd.read_csv(path)
    df = df[df['age'] >= 60]

    # Definicja cech i kolumny docelowej
    features = ['age', 'gender', 'restingrelectro', 'maxheartrate', 'oldpeak', 'slope']
    X = df[features]
    y = df['target']

    # Skalowanie tylko cech
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Wybór najlepszych cech
    selected_indices = select(pd.DataFrame(X_scaled, columns=features), y, features)
    X_selected = pd.DataFrame(X_scaled, columns=features).iloc[:, selected_indices]  # Używaj tylko wybranych cech

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Tworzenie modelu XGBoost
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)

    # Predykcja na danych testowych
    y_pred = xgb.predict(X_test)
    print("Predictions:", y_pred)

    # Porównanie z rzeczywistymi wartościami
    print("Actual values:", y_test.values)

    # Obliczanie dokładności
    accuracy = accuracy_score(y_test, y_pred)
    print("Dokładność:", accuracy)

    # Macierz pomyłek
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Macierz pomyłek:\n", conf_matrix)

if __name__ == "__main__":
    AiModel()
