import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Simulate data (replace this with real data from your watch)
data = pd.read_csv("Cardiovascular_Disease_Dataset.csv")
print(data.head())

X = data[['chestpain', 'maxheartrate']]  # Features
y = data['target']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)  # Use 3 nearest neighbors
knn.fit(X_train, y_train)

# Predict on test data
y_pred = knn.predict(X_test)
print("Predictions:", y_pred)

# Compare with actual values
print("Actual values:", y_test.values)