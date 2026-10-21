import pandas as pd
import numpy as np
import dill
import os

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv('./New_Dataset/owid-covid-data.csv')

# Keep only required columns
data = data[['date', 'location', 'new_cases']]

# Convert date
data['date'] = pd.to_datetime(data['date'])

# Rename target to match model
data.rename(columns={'new_cases': 'zika_cases'}, inplace=True)

# Fill missing values
data['zika_cases'] = data['zika_cases'].fillna(0)

# -----------------------------
# SIMPLE FEATURE ENGINEERING
# -----------------------------

# Sort data
data = data.sort_values(['location', 'date'])

# Create lag features (VERY IMPORTANT)
data['lag_1'] = data.groupby('location')['zika_cases'].shift(1)
data['lag_2'] = data.groupby('location')['zika_cases'].shift(2)
data['lag_3'] = data.groupby('location')['zika_cases'].shift(3)

# Rolling mean
data['rolling_mean_3'] = data.groupby('location')['zika_cases'].transform(lambda x: x.rolling(3).mean())

# Drop NaN rows
data = data.dropna()

# -----------------------------
# CREATE X AND y
# -----------------------------
features = ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3']

X = data[features].values
y = (data['zika_cases'] > 0).astype(int).values  # classification

# -----------------------------
# SAVE FILES (IMPORTANT)
# -----------------------------
os.makedirs('./pkl', exist_ok=True)

with open('./pkl/X.pkl', 'wb') as f:
    dill.dump(X, f)

with open('./pkl/y.pkl', 'wb') as f:
    dill.dump(y, f)

print("✅ Preprocessing done. X.pkl and y.pkl created.")