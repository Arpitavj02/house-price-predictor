"""
train_model.py
Jaipur House Price Predictor — grounded in real Book1.xlsx market data
Run once: python train_model.py  →  generates model.pkl + jaipur_property_data.csv
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

np.random.seed(42)

# ── Base price-per-sqft from Book1.xlsx (May 2026, age=1 baseline) ───────────
# Source: real Jaipur property listings — Plot and Apartment rates per locality
locality_data = {
    'Adarsh Nagar':         {'Apartment': 7650,  'Plot': 9500},
    'Agra Road':            {'Plot': 3850,        'Apartment': 3200},
    'Ajmer Road':           {'Apartment': 4550,   'Plot': 3600},
    'Ashok Nagar':          {'Plot': 20950,       'Apartment': 16000},
    'Bagru':                {'Plot': 3300,        'Apartment': 2800},
    'Bani Park':            {'Apartment': 8600,   'Plot': 10200},
    'Bapu Nagar':           {'Apartment': 13000,  'Plot': 23600},
    'Bhankrota':            {'Apartment': 4200,   'Plot': 5100},
    'Bindayaka':            {'Plot': 4500,        'Apartment': 3800},
    'C Scheme':             {'Plot': 22950,       'Apartment': 12500},
    'Chandrakala Colony':   {'Plot': 22000,       'Apartment': 17000},
    'Dholai':               {'Apartment': 4450,   'Plot': 7200},
    'Diggi Road':           {'Plot': 2350,        'Apartment': 2000},
    'Durgapura':            {'Apartment': 11150,  'Plot': 19700},
    'Gandhi Path':          {'Plot': 7650,        'Apartment': 4950},
    'Hecarawala':           {'Apartment': 4500,   'Plot': 3800},
    'Heerapura':            {'Apartment': 4450,   'Plot': 3700},
    'JLN Marg':             {'Apartment': 19050,  'Plot': 22200},
    'Jagatpura':            {'Apartment': 4900,   'Plot': 7800},
    'Jaisinghpura':         {'Plot': 5000,        'Apartment': 3900},
    'Jhotwara':             {'Apartment': 4050,   'Plot': 6850},
    'Kalwar Road':          {'Apartment': 3950,   'Plot': 3650},
    'Kanakpura':            {'Apartment': 3800,   'Plot': 3200},
    'Kardani Ka Barh':      {'Apartment': 5350,   'Plot': 4500},
    'Karola':               {'Plot': 3050,        'Apartment': 2600},
    'Lalarpura':            {'Plot': 7650,        'Apartment': 4150},
    'Mahapura':             {'Apartment': 4650,   'Plot': 4850},
    'Mahindra SEZ':         {'Apartment': 4300,   'Plot': 3800},
    'Malviya Nagar':        {'Plot': 13050,       'Apartment': 10000},
    'Mansarovar':           {'Apartment': 5300,   'Plot': 7200},
    'Mansarovar Ext.':      {'Plot': 7200,        'Apartment': 4900},
    'Manyawas':             {'Plot': 12800,       'Apartment': 9500},
    'Mata Colony':          {'Plot': 22000,       'Apartment': 17000},
    'Mohanpura':            {'Apartment': 4550,   'Plot': 3800},
    'Muhana':               {'Plot': 3000,        'Apartment': 2600},
    'Murlipura':            {'Apartment': 4350,   'Plot': 6200},
    'Narayan Vihar':        {'Plot': 7350,        'Apartment': 5100},
    'Nirman Nagar':         {'Apartment': 7750,   'Plot': 15000},
    'Patrakar Colony':      {'Plot': 6950,        'Apartment': 4250},
    'Pratap Nagar':         {'Apartment': 4700,   'Plot': 6300},
    'Raja Park':            {'Apartment': 6950,   'Plot': 8500},
    'Renwal Manji':         {'Plot': 2000,        'Apartment': 1700},
    'Ring Road':            {'Plot': 5000,        'Apartment': 4200},
    'Sanganer':             {'Apartment': 4500,   'Plot': 2950},
    'Shivdaspura':          {'Plot': 3300,        'Apartment': 2800},
    'Shyam Nagar':          {'Apartment': 9000,   'Plot': 17550},
    'Siddharth Nagar':      {'Apartment': 15000,  'Plot': 10300},
    'Sikar Road':           {'Apartment': 5200,   'Plot': 4400},
    'Sirsi Road':           {'Plot': 4850,        'Apartment': 4500},
    'Tilak Nagar':          {'Apartment': 8700,   'Plot': 20600},
    'Vaishali Nagar':       {'Apartment': 4950,   'Plot': 6500},
    'Vaishali Nagar Ext.':  {'Plot': 7650,        'Apartment': 4200},
    'Vatika':               {'Plot': 2300,        'Apartment': 1950},
    'Vatika Road':          {'Plot': 3300,        'Apartment': 2800},
    'Vidyadhar Nagar':      {'Apartment': 6500,   'Plot': 5500},
}

# Typical area ranges (sqft) by property type × BHK
AREA_MAP = {
    ('Plot', 1): (500,  900),  ('Plot', 2): (800,  1500),
    ('Plot', 3): (1200, 2200), ('Plot', 4): (1800, 3500), ('Plot', 5): (2800, 5000),
    ('Apartment', 1): (350, 650),   ('Apartment', 2): (650, 1100),
    ('Apartment', 3): (950, 1600),  ('Apartment', 4): (1400, 2200),
    ('Apartment', 5): (2000, 3500),
}

# ── Generate realistic records ────────────────────────────────────────────────
records = []
for locality, type_rates in locality_data.items():
    for prop_type, base_ppsf in type_rates.items():
        for _ in range(12):
            bhk = np.random.choice([1, 2, 3, 4, 5], p=[0.08, 0.25, 0.40, 0.20, 0.07])
            area_min, area_max = AREA_MAP[(prop_type, bhk)]
            area = np.random.randint(area_min, area_max + 1)

            age = np.random.choice(
                [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20],
                p=[0.12, 0.10, 0.12, 0.10, 0.10, 0.08, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04]
            )

            # Depreciation: apartments depreciate faster than land/plots
            dep_rate   = 0.006 if prop_type == 'Apartment' else 0.003
            age_factor = max(0.70, 1 - dep_rate * age)

            floor = 0 if prop_type == 'Plot' else np.random.randint(0, 16)
            if prop_type == 'Apartment':
                if floor == 0:      floor_factor = 0.97
                elif floor <= 4:    floor_factor = 1.00
                elif floor <= 12:   floor_factor = 1.02
                else:               floor_factor = 1.04
            else:
                floor_factor = 1.0

            furnishing        = np.random.choice([0, 1, 2], p=[0.30, 0.45, 0.25])
            furnishing_premium = {0: 0, 1: 180000, 2: 420000}[furnishing]

            has_parking      = 1 if prop_type == 'Plot' else np.random.randint(0, 2)
            parking_premium  = 180000 if has_parking else 0

            bathrooms = min(bhk, max(1, bhk - 1 + np.random.randint(0, 2)))

            effective_ppsf = base_ppsf * age_factor * floor_factor
            base_price     = effective_ppsf * area
            noise          = np.random.normal(0, base_price * 0.05)
            price          = max(base_price + furnishing_premium + parking_premium + noise, 500000)
            price          = round(price / 1000) * 1000  # round to nearest ₹1000

            records.append({
                'locality':      locality,
                'property_type': prop_type,
                'bhk':           bhk,
                'bathrooms':     bathrooms,
                'area_sqft':     area,
                'age_years':     age,
                'floor_number':  floor,
                'furnishing':    furnishing,
                'has_parking':   has_parking,
                'price':         int(price),
            })

df = pd.DataFrame(records)
print(f"Dataset: {df.shape[0]} records, {df['locality'].nunique()} localities")

# ── Encode + train ────────────────────────────────────────────────────────────
le_loc = LabelEncoder()
le_pt  = LabelEncoder()
df['locality_enc']  = le_loc.fit_transform(df['locality'])
df['prop_type_enc'] = le_pt.fit_transform(df['property_type'])

features = ['locality_enc', 'prop_type_enc', 'bhk', 'bathrooms',
            'area_sqft', 'age_years', 'floor_number', 'furnishing', 'has_parking']
X = df[features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05,
    max_depth=5, min_samples_leaf=5, random_state=42
)
model.fit(X_train, y_train)

y_pred         = model.predict(X_test)
y_pred_clamped = np.maximum(y_pred, 500000)

print(f"R² Score : {r2_score(y_test, y_pred_clamped):.4f}")
print(f"MAE      : ₹{mean_absolute_error(y_test, y_pred_clamped):,.0f}")
print(f"Min pred : ₹{y_pred_clamped.min():,.0f}")
print(f"Max pred : ₹{y_pred_clamped.max():,.0f}")
print(f"Negative : {(y_pred < 0).sum()}")

# Avg ppsf per locality for UI hint
locality_avg_ppsf = {}
for loc in sorted(df['locality'].unique()):
    sub = df[df['locality'] == loc]
    locality_avg_ppsf[loc] = round((sub['price'] / sub['area_sqft']).mean())

with open('model.pkl', 'wb') as f:
    pickle.dump({
        'model':             model,
        'le_loc':            le_loc,
        'le_pt':             le_pt,
        'features':          features,
        'locality_avg_ppsf': locality_avg_ppsf,
    }, f)

df.to_csv('jaipur_property_data.csv', index=False)
print("\n✅ model.pkl and jaipur_property_data.csv saved!")