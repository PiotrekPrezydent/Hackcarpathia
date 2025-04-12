import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import os
from sklearn.metrics import accuracy_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def select(X, y, features):
    # Selekcja cech
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    scores = selector.scores_

    # Wynik
    for f, s in zip(features, scores):
        print(f"{f}: {s:.2f}")

def AiModel():
    path = os.path.join(BASE_DIR, f"datasets/Cardiovascular_Disease_Dataset.csv")
    print(path)

    # Simulate data (replace this with real data from your watch)
    df = pd.read_csv(path)

    features = ['restingrelectro', 'maxheartrate', 'slope']
    X = df[features]
    y = df['target']

    select(X, y, features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3)  # Use 3 nearest neighbors
    knn.fit(X_train, y_train)

    # Predict on test data
    y_pred = knn.predict(X_test)
    print("Predictions:", y_pred)

    # Compare with actual values
    print("Actual values:", y_test.values)

    # Oblicz dokładność
    accuracy = accuracy_score(y_test, y_pred)
    print("Dokładność:", accuracy)

    # Macierz pomyłek
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Macierz pomyłek:\n", conf_matrix)

if __name__ == "__main__":
    AiModel()