from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    bundle = pickle.load(f)

model  = bundle['model']
scaler = bundle['scaler']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array([[
            float(data['area_sqft']),
            int(data['bedrooms']),
            int(data['bathrooms']),
            int(data['age_years']),
            float(data['distance_km']),
            int(data['has_garage']),
            int(data['has_garden']),
        ]])
        features_scaled = scaler.transform(features)
        price = model.predict(features_scaled)[0]
        return jsonify({'price': round(float(price), 2), 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

if __name__ == '__main__':
    app.run(debug=True)