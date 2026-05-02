from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import shap
import os

app = Flask(__name__)
CORS(app)

# Lazy loading — models loaded on first request
model_duration = None
model_budget = None
explainer_duration = None
encoders = None

def load_models():
    global model_duration, model_budget, explainer_duration, encoders
    if model_duration is None:
        model_duration = joblib.load('model_duration.pkl')
        model_budget = joblib.load('model_budget.pkl')
        explainer_duration = joblib.load('explainer_duration.pkl')
        encoders = joblib.load('encoders.pkl')

DEFAULTS = {
    'capacity_mw': 200,
    'planned_duration': 52,
    'planned_budget': 1000,
    'approval_delay': 8,
    'vendor_performance': 6.5,
    'vendor_past_projects': 15,
    'vendor_avg_delay': 4,
    'labour_availability': 6,
    'labour_cost_index': 1.1,
    'weather_index': 5,
    'distance_supply_hub': 200,
    'logistics_cost_index': 1.3,
    'num_contractors': 4,
    'historical_delay_rate': 20,
    'historical_risk_score': 5,
    'supply_demand_ratio': 1.0,
    'resource_availability': 6
}

def safe_float(val):
    try:
        if val is None or val == '' or val == 'null':
            return None
        return float(val)
    except (ValueError, TypeError):
        return None

@app.route('/predict', methods=['POST'])
def predict():
    load_models()
    data = request.json

    missing_count = 0
    total_fields = len(DEFAULTS)

    def get_val(key):
        nonlocal missing_count
        val = safe_float(data.get(key))
        if val is None:
            missing_count += 1
            return DEFAULTS[key]
        return val

    project_type = encoders['Project_Type'].transform([data.get('project_type', 'Substation')])[0]
    terrain_type = encoders['Terrain_Type'].transform([data.get('terrain_type', 'Flat')])[0]
    region = encoders['Region'].transform([data.get('region', 'North')])[0]
    material_volatility = encoders['Material_Cost_Volatility'].transform([data.get('material_volatility', 'Medium')])[0]

    features = np.array([[
        project_type,
        terrain_type,
        region,
        get_val('capacity_mw'),
        get_val('planned_duration'),
        get_val('planned_budget'),
        get_val('approval_delay'),
        get_val('vendor_performance'),
        get_val('vendor_past_projects'),
        get_val('vendor_avg_delay'),
        get_val('labour_availability'),
        get_val('labour_cost_index'),
        material_volatility,
        get_val('weather_index'),
        get_val('distance_supply_hub'),
        get_val('logistics_cost_index'),
        get_val('num_contractors'),
        get_val('historical_delay_rate'),
        get_val('historical_risk_score'),
        get_val('supply_demand_ratio'),
        get_val('resource_availability')
    ]])

    predicted_duration = model_duration.predict(features)[0]
    predicted_budget = model_budget.predict(features)[0]

    confidence = round(100 - (missing_count / total_fields) * 40)

    shap_values = explainer_duration.shap_values(features)
    feature_names = [
        'Project Type', 'Terrain Type', 'Region', 'Capacity MW',
        'Planned Duration', 'Planned Budget', 'Approval Delay',
        'Vendor Performance', 'Vendor Past Projects', 'Vendor Avg Delay',
        'Labour Availability', 'Labour Cost Index', 'Material Volatility',
        'Weather Index', 'Distance Supply Hub', 'Logistics Cost Index',
        'Num Contractors', 'Historical Delay Rate', 'Historical Risk Score',
        'Supply Demand Ratio', 'Resource Availability'
    ]

    shap_importance = list(zip(feature_names, np.abs(shap_values[0])))
    shap_importance.sort(key=lambda x: x[1], reverse=True)
    top_hotspots = shap_importance[:5]

    hotspots = []
    total = sum([x[1] for x in shap_importance])
    for factor, impact in top_hotspots:
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
        'confidence_score': confidence,
        'missing_fields': missing_count,
        'hotspots': hotspots
    })

@app.route('/')
def home():
    return "InfraPredict AI Backend is running!"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
