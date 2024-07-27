from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained models
regressor = joblib.load('regressor_model.pkl')
classifier = joblib.load('classifier_model.pkl')

# Define the features
features = ['condition_score', 'criticality_score', 'usage', 'failure_history', 'age', 'environment_factor']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = pd.DataFrame([data], columns=features)
    
    # Predict maintenance cost
    predicted_cost = regressor.predict(input_data)[0]
    
    # Predict priority
    predicted_priority = classifier.predict(input_data)[0]
    
    return jsonify({
        'predicted_cost': predicted_cost,
        'predicted_priority': int(predicted_priority)
    })

if __name__ == '__main__':
    app.run(debug=True)
