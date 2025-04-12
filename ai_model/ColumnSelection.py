from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

# Załaduj dane
df = pd.read_csv('Cardiovascular_Disease_Dataset.csv')  # lub jak się nazywa Twój plik

# Wybierz interesujące cechy (ze smartwatcha)
features = ['age', 'gender', 'restingrelectro', 'maxheartrate', 'oldpeak', 'slope']
X = df[features]
y = df['target']

# Selekcja cech
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X, y)
scores = selector.scores_

# Wynik
for f, s in zip(features, scores):
    print(f"{f}: {s:.2f}")
