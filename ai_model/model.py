import numpy as np
import pandas as pd

# Simulate data (replace this with real data from your watch)
data = {
    'heart_rate': [72, 75, 80, 85, 90, 95, 100, 105, 110, 115],
    'body_temperature': [36.5, 36.6, 36.7, 36.8, 37.0, 37.2, 37.5, 38.0, 38.5, 39.0],
    'condition': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0=healthy, 1=ill
}

df = pd.DataFrame(data)
print(df)