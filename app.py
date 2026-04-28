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
encoders = joblib.load('encoders.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Encode text inputs using separate encoders
    project_type = encoders['Project_Type'].transform([data['project_type']])[0]
    terrain_type = encoders['Terrain_Type'].transform([data['terrain_type']])[0]
    region = encoders['Region'].transform([data['region']])[0]
    material_volatility = encoders['Material_Cost_Volatility'].transform([data['material_volatility']])[0]

    # Build input array
    features = np.array([[
        project_type,
        terrain_type,
        region,
        float(data['planned_duration']),
        float(data['planned_budget']),
        float(data['approval_delay']),
        float(data['vendor_performance']),
        float(data['vendor_past_projects']),
        float(data['vendor_avg_delay']),
        float(data['labour_availability']),
        float(data['labour_cost_index']),
        material_volatility,
        float(data['weather_index']),
        float(data['distance_supply_hub']),
        float(data['logistics_cost_index']),
        float(data['num_contractors']),
        float(data['historical_delay_rate']),
        float(data['historical_risk_score']),
        float(data['supply_demand_ratio']),
        float(data['resource_availability'])
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
        pct = round(float((impact / total) * 100), 1)
        level = 'HIGH' if pct > 25 else 'MED' if pct > 10 else 'LOW'
        hotspots.append({
            'factor': factor,
            'impact': f'{pct:.1f}%',
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
