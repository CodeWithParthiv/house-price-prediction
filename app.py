from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model, scaler, and feature names
model = joblib.load('house_price_model.joblib')
scaler = joblib.load('scaler.joblib')
feature_names = joblib.load('feature_names.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        features = {}
        for feature in feature_names:
            value = request.form.get(feature)
            if feature in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                          'airconditioning', 'prefarea']:
                features[feature] = 1 if value == 'yes' else 0
            elif feature == 'furnishingstatus':
                features[feature] = {'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0}[value]
            else:
                features[feature] = float(value)

        # Create feature array in the correct order
        feature_array = np.array([features[feature] for feature in feature_names]).reshape(1, -1)
        
        # Scale the features
        feature_array_scaled = scaler.transform(feature_array)
        
        # Make prediction
        prediction = model.predict(feature_array_scaled)[0]
        
        return jsonify({
            'success': True,
            'predicted_price': f"₹{prediction:,.2f}"
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True) 