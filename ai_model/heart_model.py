import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models/heartd_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")  # Save scaler separately

class HeartModel:
    def __init__(self):
        self.xgb = None
        self.scaler = None
        self.selected_features = None

    def AiModel(self):
        path = os.path.join(BASE_DIR, "datasets/Cardiovascular_Disease_Dataset.csv")
        df = pd.read_csv(path)
        df = df[df['age'] >= 50]
        
        # Check class distribution
        print("Class distribution:\n", df['target'].value_counts())

        features = ['age', 'gender', 'restingrelectro', 'maxheartrate', 'oldpeak', 'slope']
        X = df[features]
        y = df['target']

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X_scaled, y)
        self.selected_features = selector.get_support(indices=True)
        print("Selected features:", [features[i] for i in self.selected_features])
        X_selected = X_scaled[:, self.selected_features]

        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

        if os.path.exists(MODEL_PATH):
            self.xgb = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            print("Model and scaler loaded from file.")
        else:
            self.xgb = XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=self.calculate_scale_pos_weight(y))
            self.xgb.fit(X_train, y_train)
            joblib.dump(self.xgb, MODEL_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            print("Model and scaler saved to file.")

        y_pred = self.xgb.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    def calculate_scale_pos_weight(self, y):
        # Calculate the ratio of negative to positive classes for imbalance adjustment
        class_counts = y.value_counts()
        return class_counts[0] / class_counts[1]

    def predict(self, input_data):
        if not hasattr(self, 'xgb') or self.xgb is None:
            raise Exception("Model nie został wytrenowany lub wczytany.")

        # Sprawdzanie, czy dane wejściowe są typu list
        if isinstance(input_data, list):
            # Konwersja listy na DataFrame
            features = ['age', 'gender', 'restingrelectro', 'maxheartrate', 'oldpeak', 'slope']
            input_df = pd.DataFrame([input_data], columns=features)
        elif isinstance(input_data, dict):
            # Jeśli dane są typu dict, konwertuj na DataFrame
            features = ['age', 'gender', 'restingrelectro', 'maxheartrate', 'oldpeak', 'slope']
            input_df = pd.DataFrame([input_data], columns=features)
        else:
            raise ValueError("Dane wejściowe muszą być typu list lub dict.")

        # Skalowanie przy użyciu zapisanej instancji skalera
        input_scaled = self.scaler.transform(input_df)

        # Wybór cech na podstawie selekcji
        input_selected = input_scaled[:, self.selected_features]

        # Dokonanie predykcji
        prediction = self.xgb.predict(input_selected)[0]
        return prediction


if __name__ == "__main__":
    model = HeartModel()
    model.AiModel()
    
    # Test prediction
    test_input = [65, 1, 1, 194, 3.7, 1]  # Example input
    prediction = model.predict(test_input)
    print(f"\nPrediction for input {test_input}: {prediction}")