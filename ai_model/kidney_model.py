import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import os
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models/kidneyd_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl") 

class KidneyModel:
    def __init__(self):
        self.xgb = None
        self.scaler = None
        self.selected_features = None

    def DatasetTransformator(self):
        # Load data
        path = os.path.join(BASE_DIR, "datasets/kidney_disease.csv")
        df_pre = pd.read_csv(path)

        # Step 1: Clean the data (fix '\t' in target)
        df_pre['target'] = df_pre['target'].str.strip()  # Remove hidden tabs/spaces

        # Step 2: Define columns by type
        binary_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']  # Yes/No or Normal/Abnormal
        nominal_cols = ['al', 'su', 'sg']  # Multi-class (no ordinal meaning)
        numerical_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']

        # Step 3: Auto-convert categorical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('binary', OrdinalEncoder(), binary_cols),  # Binary → 0/1
                ('nominal', OneHotEncoder(drop='first'), nominal_cols),  # Nominal → One-hot
                ('numeric', 'passthrough', numerical_cols)  # Leave numbers unchanged
            ])

        # Apply preprocessing
        X_processed = preprocessor.fit_transform(df_pre)

        # Convert back to DataFrame (optional)
        feature_names = (
            binary_cols + 
            list(preprocessor.named_transformers_['nominal'].get_feature_names_out(nominal_cols)) +
            numerical_cols
        )
        df_processed = pd.DataFrame(X_processed, columns=feature_names)

        # Step 4: Encode target
        df_processed['target'] = df_pre['target'].map({'ckd': 1, 'notckd': 0})

        return df_processed

    def AiModel(self):
        df = self.DatasetTransformator()
        print(df.head())

        df = df[df['age'] >= 50]

        print("Class distribution:\n", df['target'].value_counts())

        features = ['bp', 'hemo', 'htn', 'age', 'dm']
        X = df[features]
        y = df['target']

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

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

if __name__ == "__main__":
    model = KidneyModel()
    model.AiModel()