from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# ── Register custom Jinja2 filter: formats an integer as Indian numbering ──────
def format_inr(value):
    """Format a number in Indian style: 1,23,456"""
    s = str(int(value))
    if len(s) <= 3:
        return s
    result = s[-3:]
    s = s[:-3]
    while len(s) > 2:
        result = s[-2:] + ',' + result
        s = s[:-2]
    if s:
        result = s + ',' + result
    return result

app.jinja_env.filters['format_inr'] = format_inr

# ── Load model bundle ──────────────────────────────────────────────────────────
with open('model.pkl', 'rb') as f:
    bundle = pickle.load(f)

model             = bundle['model']
le_loc            = bundle['le_loc']
le_pt             = bundle['le_pt']
features          = bundle['features']
locality_avg_ppsf = bundle['locality_avg_ppsf']

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    localities = sorted(le_loc.classes_.tolist())
    return render_template('index.html',
                           localities=localities,
                           locality_avg_ppsf=locality_avg_ppsf)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        d = request.get_json()

        locality_enc  = int(le_loc.transform([d['locality']])[0])
        prop_type_enc = int(le_pt.transform([d['property_type']])[0])

        row = pd.DataFrame([[
            locality_enc,
            prop_type_enc,
            int(d['bhk']),
            int(d['bathrooms']),
            float(d['area_sqft']),
            int(d['age_years']),
            int(d['floor_number']),
            int(d['furnishing']),
            int(d['has_parking']),
        ]], columns=features)

        price = float(model.predict(row)[0])
        price = max(price, 500000)        # hard floor ₹5 lakh

        price_per_sqft = round(price / float(d['area_sqft']))

        return jsonify({
            'status':          'success',
            'price':           round(price, -3),   # round to nearest ₹1000
            'price_per_sqft':  price_per_sqft,
            'market_avg_ppsf': locality_avg_ppsf.get(d['locality'], 0),
        })

    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)