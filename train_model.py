import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import pickle

# Sample dataset — replace with real data if you have it
np.random.seed(42)
n = 500

data = pd.DataFrame({
    'area_sqft':      np.random.randint(500, 5000, n),
    'bedrooms':       np.random.randint(1, 6, n),
    'bathrooms':      np.random.randint(1, 4, n),
    'age_years':      np.random.randint(0, 40, n),
    'distance_km':    np.random.uniform(1, 30, n),
    'has_garage':     np.random.randint(0, 2, n),
    'has_garden':     np.random.randint(0, 2, n),
})

# Synthetic price formula
data['price'] = (
    data['area_sqft'] * 180
    + data['bedrooms'] * 50000
    + data['bathrooms'] * 30000
    - data['age_years'] * 8000
    - data['distance_km'] * 15000
    + data['has_garage'] * 120000
    + data['has_garden'] * 80000
    + np.random.normal(0, 80000, n)
)

X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(f"R² Score : {r2_score(y_test, y_pred):.4f}")
print(f"MAE      : ₹{mean_absolute_error(y_test, y_pred):,.0f}")

# Save model + scaler together
with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)

print("✅ model.pkl saved!")