import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os
import joblib  # Biblioteka do zapisywania i wczytywania modelu

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model.pkl")  # Ścieżka do zapisanego modelu

class HeartModel:
    def __init__(self):
        self.xgb = None

    def AiModel(self):
        path = os.path.join(BASE_DIR, f"datasets/Cardiovascular_Disease_Dataset.csv")
        print(path)

        df = pd.read_csv(path)
        df = df[df['age'] >= 60]

        features = ['age', 'gender', 'restingrelectro', 'maxheartrate', 'oldpeak', 'slope']
        X = df[features]
        y = df['target']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        selected_indices = self.select(pd.DataFrame(X_scaled, columns=features), y, features)
        X_selected = pd.DataFrame(X_scaled, columns=features).iloc[:, selected_indices]

        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

        if os.path.exists(MODEL_PATH):
            # Wczytaj istniejący model
            self.xgb = joblib.load(MODEL_PATH)
            print("Model wczytany z pliku.")
        else:
            # Utwórz i wytrenuj model, a następnie zapisz go do pliku
            self.xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            self.xgb.fit(X_train, y_train)
            joblib.dump(self.xgb, MODEL_PATH)
            print("Model zapisany do pliku.")

        y_pred = self.xgb.predict(X_test)
        print("Predictions:", y_pred)
        print("Actual values:", y_test.values)
        accuracy = accuracy_score(y_test, y_pred)
        print("Dokładność:", accuracy)

        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Macierz pomyłek:\n", conf_matrix)

    def select(self, X, y, features):
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        scores = selector.scores_
        selected_features = selector.get_support(indices=True)
        print("Wybrane cechy:", [features[i] for i in selected_features])
        return selected_features

    def predict(self, data):
        return self.xgb.predict(data)

if __name__ == "__main__":
    model = HeartModel()
    model.AiModel()

