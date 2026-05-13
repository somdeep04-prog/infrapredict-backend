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
model_material = None
model_logistics = None
model_labour = None
explainer_duration = None
encoders = None
 
def load_models():
    global model_duration, model_budget, model_material, model_logistics, model_labour, explainer_duration, encoders
    if model_duration is None:
        model_duration = joblib.load('model_duration.pkl')
        model_budget = joblib.load('model_budget.pkl')
        model_material = joblib.load('model_material.pkl')
        model_logistics = joblib.load('model_logistics.pkl')
        model_labour = joblib.load('model_labour.pkl')
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
 
TERRAIN_MAP    = {'Flat': 1, 'Urban': 2, 'Hilly': 3, 'Mountainous': 4}
VOLATILITY_MAP = {'Low': 1, 'Medium': 2, 'High': 3}
REGION_MAP     = {'South': 1, 'West': 2, 'North': 3, 'East': 4}
PROJECT_MAP    = {'Substation': 1, 'Distribution Line': 2, 'Underground Cable': 3,
                  'Solar Power Plant': 4, 'Wind Power Plant': 5,
                  'Transmission Line': 6, 'HVDC Line': 7}
 
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
 
    project_type        = encoders['Project_Type'].transform([data.get('project_type', 'Substation')])[0]
    terrain_type        = encoders['Terrain_Type'].transform([data.get('terrain_type', 'Flat')])[0]
    region              = encoders['Region'].transform([data.get('region', 'North')])[0]
    material_volatility = encoders['Material_Cost_Volatility'].transform([data.get('material_volatility', 'Medium')])[0]
 
    # Raw values
    capacity_mw           = get_val('capacity_mw')
    planned_duration      = get_val('planned_duration')
    planned_budget        = get_val('planned_budget')
    approval_delay        = get_val('approval_delay')
    vendor_performance    = get_val('vendor_performance')
    vendor_past_projects  = get_val('vendor_past_projects')
    vendor_avg_delay      = get_val('vendor_avg_delay')
    labour_availability   = get_val('labour_availability')
    labour_cost_index     = get_val('labour_cost_index')
    weather_index         = get_val('weather_index')
    distance_supply_hub   = get_val('distance_supply_hub')
    logistics_cost_index  = get_val('logistics_cost_index')
    num_contractors       = get_val('num_contractors')
    historical_delay_rate = get_val('historical_delay_rate')
    historical_risk_score = get_val('historical_risk_score')
    supply_demand_ratio   = get_val('supply_demand_ratio')
    resource_availability = get_val('resource_availability')
 
    # Original 21-feature array — duration & budget models unchanged
    features = np.array([[
        project_type, terrain_type, region,
        capacity_mw, planned_duration, planned_budget,
        approval_delay, vendor_performance, vendor_past_projects, vendor_avg_delay,
        labour_availability, labour_cost_index, material_volatility,
        weather_index, distance_supply_hub, logistics_cost_index,
        num_contractors, historical_delay_rate, historical_risk_score,
        supply_demand_ratio, resource_availability
    ]])
 
    predicted_duration = model_duration.predict(features)[0]
    predicted_budget   = model_budget.predict(features)[0]
 
    confidence  = round(100 - (missing_count / total_fields) * 40)
 
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
        pct   = round(float((impact / total) * 100), 1)
        level = 'HIGH' if pct > 25 else 'MED' if pct > 10 else 'LOW'
        hotspots.append({'factor': factor, 'impact': f'{pct:.1f}%', 'level': level})
 
    # Derived features — for material, logistics, labour models
    terrain_score    = TERRAIN_MAP.get(data.get('terrain_type', 'Flat'), 1)
    volatility_score = VOLATILITY_MAP.get(data.get('material_volatility', 'Medium'), 2)
    project_score    = PROJECT_MAP.get(data.get('project_type', 'Substation'), 1)
 
    vendor_risk_score            = ((10 - vendor_performance) + vendor_avg_delay) / 2
    logistics_pressure           = (distance_supply_hub / 100) * logistics_cost_index
    labour_stress                = (10 - labour_availability) * labour_cost_index
    weather_terrain_risk         = weather_index * terrain_score
    supply_chain_risk            = (distance_supply_hub / 100) + (1 - supply_demand_ratio) * 10 + volatility_score * 2
    project_complexity           = (capacity_mw / 100) * terrain_score * project_score
    historical_performance_index = historical_delay_rate * historical_risk_score
    resource_stress              = (10 - resource_availability) * (1 / (supply_demand_ratio + 0.01))
    budget_per_mw                = planned_budget / (capacity_mw + 1)
    duration_per_mw              = planned_duration / (capacity_mw + 1)
    contractor_load              = num_contractors / (capacity_mw + 1)
    approval_impact              = approval_delay / (planned_duration + 1)
    overall_risk_index           = (vendor_risk_score * 0.25 + weather_terrain_risk * 0.20 +
                                    supply_chain_risk * 0.20 + historical_performance_index * 0.20 +
                                    resource_stress * 0.15)
    cost_pressure_index          = (labour_stress * 0.3 + logistics_pressure * 0.3 +
                                    volatility_score * 3 * 0.2 + supply_chain_risk * 0.2)
    delay_risk_score             = (approval_delay * 0.25 + vendor_risk_score * 0.25 +
                                    weather_terrain_risk * 0.25 + historical_performance_index * 0.25)
 
    # 36-feature array — material, logistics, labour models
    features_extended = np.array([[
        project_type, terrain_type, region,
        capacity_mw, planned_duration, planned_budget,
        approval_delay, vendor_performance, vendor_past_projects, vendor_avg_delay,
        labour_availability, labour_cost_index, material_volatility,
        weather_index, distance_supply_hub, logistics_cost_index,
        num_contractors, historical_delay_rate, historical_risk_score,
        supply_demand_ratio, resource_availability,
        vendor_risk_score, logistics_pressure, labour_stress,
        weather_terrain_risk, supply_chain_risk, project_complexity,
        historical_performance_index, resource_stress,
        budget_per_mw, duration_per_mw, contractor_load,
        approval_impact, overall_risk_index, cost_pressure_index,
        delay_risk_score
    ]])
 
    predicted_material  = float(model_material.predict(features_extended)[0])
    predicted_logistics = float(model_logistics.predict(features_extended)[0])
    predicted_labour    = float(model_labour.predict(features_extended)[0])
 
    return jsonify({
        'predicted_duration' : round(float(predicted_duration)),
        'predicted_budget'   : round(float(predicted_budget)),
        'predicted_material' : round(predicted_material, 1),
        'predicted_logistics': round(predicted_logistics, 1),
        'predicted_labour'   : round(predicted_labour, 1),
        'confidence_min'     : round(float(predicted_duration) * 0.9),
        'confidence_max'     : round(float(predicted_duration) * 1.1),
        'confidence_score'   : confidence,
        'missing_fields'     : missing_count,
        'hotspots'           : hotspots
    })
 
@app.route('/')
def home():
    return "InfraPredict AI Backend is running!"
 
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
