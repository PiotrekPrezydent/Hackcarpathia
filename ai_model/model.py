import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Simulate data (replace this with real data from your watch)
data = {
    'heart_rate': [72, 75, 80, 85, 90, 95, 100, 105, 110, 115],
    'body_temperature': [36.5, 36.6, 36.7, 36.8, 37.0, 37.2, 37.5, 38.0, 38.5, 39.0],
    'condition': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0=healthy, 1=ill
}

df = pd.DataFrame(data)
print(df)

X = df[['heart_rate', 'body_temperature']]  # Features
y = df['condition']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)  # Use 3 nearest neighbors
knn.fit(X_train, y_train)

# Predict on test data
y_pred = knn.predict(X_test)
print("Predictions:", y_pred)

# Compare with actual values
print("Actual values:", y_test.values)