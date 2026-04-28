from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import shap

app = Flask(__name__)
CORS(app)

# Load all saved models
model_duration = joblib.load('model_duration.pkl')
model_budget = joblib.load('model_budget.pkl')
explainer_duration = joblib.load('explainer_duration.pkl')
le = joblib.load('label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Encode text inputs
    project_type = le.transform([data['project_type']])[0]
    terrain_type = le.transform([data['terrain_type']])[0]
    region = le.transform([data['region']])[0]
    material_volatility = le.transform([data['material_volatility']])[0]

    # Build input array
    features = np.array([[
        project_type,
        terrain_type,
        region,
        data['planned_duration'],
        data['planned_budget'],
        data['approval_delay'],
        data['vendor_performance'],
        data['vendor_past_projects'],
        data['vendor_avg_delay'],
        data['labour_availability'],
        data['labour_cost_index'],
        material_volatility,
        data['weather_index'],
        data['distance_supply_hub'],
        data['logistics_cost_index'],
        data['num_contractors'],
        data['historical_delay_rate'],
        data['historical_risk_score'],
        data['supply_demand_ratio'],
        data['resource_availability']
    ]])

    # Predictions
    predicted_duration = model_duration.predict(features)[0]
    predicted_budget = model_budget.predict(features)[0]

    # SHAP hotspots
    shap_values = explainer_duration.shap_values(features)
    feature_names = [
        'Project Type', 'Terrain Type', 'Region',
        'Planned Duration', 'Planned Budget', 'Approval Delay',
        'Vendor Performance', 'Vendor Past Projects', 'Vendor Avg Delay',
        'Labour Availability', 'Labour Cost Index', 'Material Volatility',
        'Weather Index', 'Distance Supply Hub', 'Logistics Cost Index',
        'Num Contractors', 'Historical Delay Rate', 'Historical Risk Score',
        'Supply Demand Ratio', 'Resource Availability'
    ]

    # Get top 5 hotspots
    shap_importance = list(zip(feature_names, np.abs(shap_values[0])))
    shap_importance.sort(key=lambda x: x[1], reverse=True)
    top_hotspots = shap_importance[:5]

    hotspots = []
    for factor, impact in top_hotspots:
        total = sum([x[1] for x in shap_importance])
        pct = round((impact / total) * 100, 1)
        level = 'HIGH' if pct > 25 else 'MED' if pct > 10 else 'LOW'
        hotspots.append({
            'factor': factor,
            'impact': f'{pct}%',
            'level': level
        })

    return jsonify({
        'predicted_duration': round(float(predicted_duration)),
        'predicted_budget': round(float(predicted_budget)),
        'confidence_min': round(float(predicted_duration) * 0.9),
        'confidence_max': round(float(predicted_duration) * 1.1),
        'hotspots': hotspots
    })

@app.route('/')
def home():
    return "InfraPredict AI Backend is running!"

if __name__ == '__main__':
    app.run(debug=True)
