"""
Flood Sentinel - Flask Backend Application
===========================================
Main Flask application providing:
- Static file serving for frontend
- REST API endpoints for flood risk data
- Live weather integration for dynamic risk assessment

Author: Flood Sentinel Team
Date: 2025
Phase: 2 - Backend Development
"""

import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
from datetime import datetime
import time
from sklearn.neighbors import BallTree
import math
import json

# Import API keys securely from environment
from config import OPENWEATHER_API_KEY, NOAA_API_KEY

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'flood-sentinel-dev-key-2025')
app.config['JSON_SORT_KEYS'] = False

# File paths
MODEL_PATH = 'models/flood_risk_model.pkl'
SCALER_PATH = 'models/feature_scaler.pkl'
DATA_PATH = 'data/final_training_dataset.csv'

# OpenWeather API configuration
OPENWEATHER_API_BASE = 'https://api.openweathermap.org/data/2.5/weather'

# Global variables for loaded assets
model = None
scaler = None
dataset = None


def load_model_and_data():
    """Load the trained model, scaler, and dataset on startup."""
    global model, scaler, dataset
    
    try:
        # Load model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Loaded flood risk model from {MODEL_PATH}")
        
        # Load scaler
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✓ Loaded feature scaler from {SCALER_PATH}")
        
        # Load dataset
        dataset = pd.read_csv(DATA_PATH)
        print(f"✓ Loaded dataset: {len(dataset)} records")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model/data: {e}")
        return False


def build_feature_matrix(df_like: pd.DataFrame) -> pd.DataFrame:
    """Create the exact feature matrix the model expects, including engineered features.

    Expected base columns in df_like:
      - elevation_m, slope_percent, distance_to_water_m,
        precipitation_mm_event, soil_permeability_mm_hr, land_cover
    """
    base_cols = [
        'elevation_m',
        'slope_percent',
        'distance_to_water_m',
        'precipitation_mm_event',
        'soil_permeability_mm_hr',
        'land_cover'
    ]
    X = df_like[base_cols].copy()
    # Engineered features (must match 3_train_model.py)
    X['elevation_precip_interaction'] = X['elevation_m'] * X['precipitation_mm_event']
    X['water_soil_interaction'] = X['distance_to_water_m'] * X['soil_permeability_mm_hr']
    X['runoff_potential'] = X['precipitation_mm_event'] / (X['soil_permeability_mm_hr'] + 1)
    return X


def predict_probabilities_with_names(X_features: pd.DataFrame) -> np.ndarray:
    """Scale features and predict probabilities while preserving feature names.

    This avoids sklearn warnings about missing feature names by reconstructing a
    DataFrame with the original column names after scaling.
    """
    # Keep original feature order/names
    col_names = list(X_features.columns)
    X_scaled = scaler.transform(X_features)
    try:
        X_scaled_df = pd.DataFrame(X_scaled, columns=col_names)
        return model.predict_proba(X_scaled_df)[:, 1]
    except Exception:
        # Fallback if model was trained without feature names
        return model.predict_proba(X_scaled)[:, 1]


def rule_based_probabilities(raw_features: pd.DataFrame, precip_in: float, *,
                             local_elev_median: float) -> np.ndarray:
    """Hydrology-physics risk model preserving heterogeneity at all rainfall levels.

    - Uses ONLY slider precipitation (inches). If precip_in == 0, risk -> ~0
    - Combines SCS CN runoff with TWI-like pooling and terrain modifiers.
    - Maintains gradients even under extreme rainfall (no uniform saturation).
    """
    P_mm = max(0.0, float(precip_in)) * 25.4

    elev = raw_features['elevation_m'].astype(float).to_numpy()
    slope_pct = raw_features['slope_percent'].astype(float).to_numpy()
    slope_rad = np.clip(np.radians(slope_pct), 1e-3, np.pi/2 - 1e-3)  # radians, avoid div0
    distw = raw_features['distance_to_water_m'].astype(float).to_numpy()
    soilp = raw_features['soil_permeability_mm_hr'].astype(float).to_numpy()
    landc = raw_features['land_cover'].astype(float).to_numpy()

    # ========== 1. SCS Curve Number Runoff ==========
    # HSG from permeability (mm/hr)
    def infer_hsg(v):
        if v >= 7.6: return 'A'
        if v >= 3.8: return 'B'
        if v >= 1.3: return 'C'
        return 'D'
    hsg = np.array([infer_hsg(v) for v in soilp])

    # CN mapping (NRCS tables for AMC II)
    def cn_for(land, g):
        tables = {
            'urban_dense': {'A':77,'B':85,'C':90,'D':92},
            'urban_res':   {'A':61,'B':75,'C':83,'D':87},
            'open_grass':  {'A':39,'B':61,'C':74,'D':80},
            'forest':      {'A':30,'B':55,'C':70,'D':77},
            'agri':        {'A':67,'B':78,'C':85,'D':89},
            'default':     {'A':68,'B':79,'C':86,'D':89},
        }
        if land >= 90: key = 'urban_dense'
        elif land >= 60: key = 'urban_res'
        elif land >= 40: key = 'open_grass'
        elif land >= 20: key = 'forest'
        elif land >= 10: key = 'agri'
        else: key = 'default'
        return tables[key][g]

    CN = np.array([cn_for(lc, g) for lc, g in zip(landc, hsg)], dtype=float)
    S = 25400.0 / np.clip(CN, 30.0, 98.0) - 254.0  # mm
    Ia = 0.2 * S
    runoff_mm = np.where(P_mm <= Ia, 0.0, ((P_mm - Ia)**2) / (P_mm + 0.8 * S + 1e-6))

    # ========== 2. Terrain-Based Risk Factors ==========
    # Elevation relative to local median (lower = higher risk)
    elev_rel = elev - float(local_elev_median)
    elev_factor = 1.0 / (1.0 + np.exp(elev_rel / 3.5))  # sigmoid, ~0.5 at median

    # Slope: flatter = higher risk (TWI-inspired)
    # Gentle saturation so slope differences remain visible
    slope_factor = 1.0 / (1.0 + np.sqrt(slope_pct + 0.5))  # 0..1, high when flat

    # Distance to water: proximity = higher risk (exponential decay)
    dist_factor = np.exp(-distw / 400.0)  # ~0.22 at 600m, ~0.08 at 1200m

    # Soil infiltration deficit: lower permeability = higher risk
    infil_factor = 1.0 / (1.0 + (soilp / 15.0))  # 0..1, high when imperv

    # Imperviousness proxy from land cover
    imperv = np.clip((landc - 55.0) / 45.0, 0.0, 1.0)  # 0..1 for codes 55–100

    # ========== 3. TWI-Like Pooling Index ==========
    # Approximate upslope contributing area from local elevation + slope
    # (Proper TWI needs flow accumulation; we approximate with elevation & flatness)
    local_flat = 1.0 - np.tanh(slope_pct / 8.0)  # 0..1
    local_low = 1.0 / (1.0 + np.exp(elev_rel / 4.0))  # 0..1, high when below median
    pooling_suscept = 0.6 * local_flat + 0.4 * local_low  # combined pooling proxy

    # ========== 4. Multi-Component Risk Model ==========
    # A. Channel/Fluvial Component (near rivers, driven by runoff)
    # Strongly weight distance to water to create sharp river gradients
    channel_base = 0.20 * elev_factor + 0.60 * dist_factor + 0.12 * slope_factor + 0.08 * infil_factor
    channel_base = np.clip(channel_base, 0.02, 1.0)
    # Scale runoff to 0..1 but preserve heterogeneity (no hard saturation)
    runoff_norm = runoff_mm / (runoff_mm + 140.0)  # ~0.64 at 254mm, gentler
    channel_risk = channel_base * runoff_norm

    # B. Pluvial/Pooling Component (anywhere water can pool)
    # Driven by rainfall intensity, flatness, imperviousness, pooling susceptibility
    rain_driver = P_mm / (P_mm + 100.0)  # ~0.72 at 254mm, preserves more gradient
    pluvial_base = 0.40 * pooling_suscept + 0.35 * slope_factor + 0.15 * infil_factor + 0.10 * imperv
    pluvial_base = np.clip(pluvial_base, 0.0, 1.0)
    pluvial_risk = pluvial_base * rain_driver

    # C. Combined Risk with Rain-Dependent Weighting
    # At low rain: channel dominates. At high rain: pluvial pooling dominates.
    w_pluvial = P_mm / (P_mm + 180.0)  # ~0.59 at 254mm - more balanced
    w_channel = 1.0 - w_pluvial
    combined_risk = w_channel * channel_risk + w_pluvial * pluvial_risk

    # ========== 5. Final Probability with Heterogeneity-Preserving Mapping ==========
    # Use a power function with gentle compression to preserve wider spreads
    # while reaching realistic high probabilities at extreme rainfall
    # At 10" rain target ranges:
    #   - High-risk (flat, low, near water): combined_risk ~0.5-0.7 → prob ~0.60-0.85
    #   - Medium-risk (moderate terrain): combined_risk ~0.3-0.5 → prob ~0.35-0.60
    #   - Low-risk (elevated, sloped, far): combined_risk ~0.1-0.3 → prob ~0.15-0.35
    prob = combined_risk ** 0.85  # Power < 1 raises probabilities while preserving spread
    return np.clip(prob, 0.0, 0.98)


def get_current_weather(lat, lon):
    """
    Fetch current weather data from OpenWeather API.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        dict: Weather data including precipitation, or None if failed
    """
    if not OPENWEATHER_API_KEY:
        print("⚠ OpenWeather API key not configured")
        return None
    
    try:
        params = {
            'lat': lat,
            'lon': lon,
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric'
        }
        
        response = requests.get(OPENWEATHER_API_BASE, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract relevant weather data
            weather_info = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'description': data['weather'][0]['description'],
                'precipitation_1h': data.get('rain', {}).get('1h', 0),  # mm in last hour
                'precipitation_3h': data.get('rain', {}).get('3h', 0),  # mm in last 3 hours
                'wind_speed': data['wind']['speed'],
                'clouds': data['clouds']['all']
            }
            
            return weather_info
        else:
            print(f"⚠ OpenWeather API error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"⚠ Error fetching weather: {e}")
        return None


def calculate_dynamic_risk(base_features, weather_data):
    """
    Calculate dynamic flood risk by adjusting base risk with current weather.
    
    Args:
        base_features: DataFrame with base environmental features
        weather_data: Current weather data dict
        
    Returns:
        dict: Risk assessment with probability and level
    """
    # Make a copy of features
    features = base_features.copy()
    
    # If we have recent precipitation, adjust the precipitation feature
    if weather_data and (weather_data['precipitation_1h'] > 0 or weather_data['precipitation_3h'] > 0):
        # Use 3-hour precipitation if available, else 1-hour
        current_precip = weather_data['precipitation_3h'] or weather_data['precipitation_1h']
        
        # Add current precipitation to the event precipitation
        # (This simulates ongoing rainfall increasing flood risk)
        features['precipitation_mm_event'] += current_precip
        
        # Recalculate engineered features that depend on precipitation
        features['elevation_precip_interaction'] = (
            features['elevation_m'] * features['precipitation_mm_event']
        )
        features['runoff_potential'] = (
            features['precipitation_mm_event'] / (features['soil_permeability_mm_hr'] + 1)
        )
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict probability
    flood_probability = model.predict_proba(features_scaled)[0][1]
    
    # Determine risk level
    if flood_probability < 0.2:
        risk_level = 'Low'
        risk_color = '#28a745'  # Green
    elif flood_probability < 0.4:
        risk_level = 'Moderate'
        risk_color = '#ffc107'  # Yellow
    elif flood_probability < 0.6:
        risk_level = 'High'
        risk_color = '#fd7e14'  # Orange
    else:
        risk_level = 'Extreme'
        risk_color = '#dc3545'  # Red
    
    return {
        'flood_probability': float(flood_probability),
        'risk_level': risk_level,
        'risk_color': risk_color,
        'weather_adjusted': weather_data is not None and (
            weather_data.get('precipitation_1h', 0) > 0 or 
            weather_data.get('precipitation_3h', 0) > 0
        )
    }


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Render the main application page with cache-busting build token."""
    return render_template('index.html', build=int(time.time()))


@app.route('/test')
def test_page():
    """Simple test page to verify server is responding."""
    return render_template('test.html')


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'dataset_loaded': dataset is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/risk-predictions')
def get_risk_predictions():
    """
    Get historical flood risk predictions for all points in dataset.
    Returns GeoJSON format for map visualization.
    """
    if dataset is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    
    try:
        # Build feature matrix and predict in one vectorized pass
        X = build_feature_matrix(dataset)
        probabilities = predict_probabilities_with_names(X)
        
        # Create GeoJSON features
        features = []
        for idx, row in dataset.iterrows():
            prob = float(probabilities[idx])
            
            # Determine risk level and color
            if prob < 0.2:
                risk_level = 'Low'
                color = '#10b981'
            elif prob < 0.4:
                risk_level = 'Moderate'
                color = '#f59e0b'
            elif prob < 0.6:
                risk_level = 'High'
                color = '#f97316'
            else:
                risk_level = 'Extreme'
                color = '#ef4444'
            
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [row['longitude'], row['latitude']]
                },
                'properties': {
                    # Keep house_id as string (e.g., "HOUSE_00001")
                    'house_id': str(row.get('house_id', idx)),
                    'flood_probability': prob,
                    'risk_level': risk_level,
                    'color': color,
                    'elevation_m': float(row.get('elevation_m', 0.0)),
                    'distance_to_water_m': float(row.get('distance_to_water_m', 0.0)),
                    'historical_flood': int(row.get('historical_flood', 0))
                }
            }
            features.append(feature)
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        return jsonify(geojson)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ----------------------------------------------------------------------------
# MUST-HAVES: Validation, Real-time Data, Forecast, Address Report
# ----------------------------------------------------------------------------

@app.route('/api/validation/metrics')
def validation_metrics():
    """Compute basic precision/recall/F1 using dataset historical_flood as proxy truth."""
    if dataset is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    try:
        thr = float(request.args.get('threshold', 0.5))
        X = build_feature_matrix(dataset)
        probs = predict_probabilities_with_names(X)
        y_true = (dataset['historical_flood'].astype(int).to_numpy() if 'historical_flood' in dataset.columns else np.zeros(len(dataset), dtype=int))
        y_pred = (probs >= thr).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float(2 * precision * recall / max(precision + recall, 1e-9))
        acc = float((tp + tn) / max(tp + fp + fn + tn, 1))
        return jsonify({
            'threshold': thr,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': acc,
            'counts': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
            'n': int(len(dataset))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/usgs-gauges', methods=['GET'])
def usgs_gauges():
    """Fetch USGS stream gauges near a point using bbox derived from radius_km."""
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        radius_km = float(request.args.get('radius_km', 30))
        meters_per_deg_lat = 111000.0
        meters_per_deg_lon = 111000.0 * max(np.cos(np.radians(lat)), 1e-6)
        dlat = (radius_km * 1000.0) / meters_per_deg_lat
        dlon = (radius_km * 1000.0) / meters_per_deg_lon
        south, north = lat - dlat, lat + dlat
        west, east = lon - dlon, lon + dlon
        url = (
            f"https://waterservices.usgs.gov/nwis/iv/?format=json&parameterCd=00065,00060&bBox={west},{south},{east},{north}"
        )
        r = requests.get(url, timeout=25)
        gauges = []
        if r.status_code == 200:
            j = r.json()
            series = j.get('value', {}).get('timeSeries', [])
            latest = {}
            meta = {}
            for ts in series:
                site = ts.get('sourceInfo', {}).get('siteCode', [{}])[0].get('value')
                if not site: continue
                variable = ts.get('variable', {}).get('variableCode', [{}])[0].get('value')
                vals = ts.get('values', [{}])[0].get('value', [])
                if not vals: continue
                val = vals[-1].get('value')
                unit = ts.get('variable', {}).get('unit', {}).get('unitCode')
                if site not in meta:
                    si = ts.get('sourceInfo', {})
                    loc = si.get('geoLocation', {}).get('geogLocation', {})
                    meta[site] = {
                        'site': site,
                        'name': si.get('siteName'),
                        'lat': loc.get('latitude'),
                        'lon': loc.get('longitude')
                    }
                latest.setdefault(site, {})[variable] = {'value': float(val), 'unit': unit}
            for site, info in meta.items():
                gh = latest.get(site, {}).get('00065', {}).get('value')  # gage height, ft
                q = latest.get(site, {}).get('00060', {}).get('value')   # discharge, cfs
                gauges.append({
                    **info,
                    'gage_height_ft': float(gh) if gh is not None else None,
                    'discharge_cfs': float(q) if q is not None else None
                })
        return jsonify({'gauges': gauges})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nws/alerts', methods=['GET'])
def nws_alerts():
    """Fetch active NWS alerts for a point."""
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        url = f"https://api.weather.gov/alerts/active?point={lat},{lon}"
        headers = {'User-Agent': 'FloodScope/1.0'}
        r = requests.get(url, headers=headers, timeout=25)
        features = []
        if r.status_code == 200:
            j = r.json()
            for f in j.get('features', [])[:50]:
                prop = f.get('properties', {})
                features.append({
                    'id': prop.get('id'),
                    'event': prop.get('event'),
                    'headline': prop.get('headline'),
                    'severity': prop.get('severity'),
                    'status': prop.get('status'),
                    'effective': prop.get('effective'),
                    'expires': prop.get('expires'),
                })
        return jsonify({'alerts': features})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _nws_qpf_timeseries(lat, lon):
    headers = {'User-Agent': 'FloodScope/1.0'}
    p = requests.get(f"https://api.weather.gov/points/{lat},{lon}", headers=headers, timeout=25)
    p.raise_for_status()
    j = p.json()
    office = j['properties']['cwa']
    gridX = j['properties']['gridX']
    gridY = j['properties']['gridY']
    g = requests.get(f"https://api.weather.gov/gridpoints/{office}/{gridX},{gridY}", headers=headers, timeout=25)
    g.raise_for_status()
    gj = g.json()
    qpf = gj['properties'].get('quantitativePrecipitation', {})
    values = qpf.get('values', [])
    series = []
    for v in values:
        # unit kg/m^2 ~ mm
        amount_mm = float(v.get('value', 0.0))
        # validTime like '2025-10-15T01:00:00+00:00/PT1H'
        vt = v.get('validTime', '')
        start = vt.split('/')[0]
        series.append({'time': start, 'mm': amount_mm, 'inches': amount_mm / 25.4})
    return series


@app.route('/api/forecast-risk', methods=['POST'])
def forecast_risk():
    """Forecast risk at a point using NWS QPF for horizons [3,6,12,24] hours."""
    if dataset is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    try:
        data = request.get_json() or {}
        lat = float(data.get('latitude'))
        lon = float(data.get('longitude'))
        horizons = [3, 6, 12, 24]

        # Nearest features
        dataset['distance_to_query'] = np.sqrt((dataset['latitude'] - lat)**2 + (dataset['longitude'] - lon)**2)
        nearest_idx = dataset['distance_to_query'].idxmin()
        nearest = dataset.iloc[nearest_idx]
        raw = pd.DataFrame([{ 
            'elevation_m': float(nearest['elevation_m']),
            'slope_percent': float(nearest['slope_percent']),
            'distance_to_water_m': float(nearest['distance_to_water_m']),
            'precipitation_mm_event': float(nearest['precipitation_mm_event']),
            'soil_permeability_mm_hr': float(nearest['soil_permeability_mm_hr']),
            'land_cover': float(nearest['land_cover'])
        }])

        # QPF series
        qpf_series = _nws_qpf_timeseries(lat, lon)
        # Aggregate to horizons
        result = []
        for h in horizons:
            inches = sum([pt['inches'] for pt in qpf_series[:h]])  # first h hours
            p = rule_based_probabilities(raw, inches, local_elev_median=float(nearest['elevation_m']))
            result.append({'hours': h, 'precip_inches': inches, 'probability': float(p[0])})

        dataset.drop('distance_to_query', axis=1, inplace=True)
        return jsonify({'point': {'lat': lat, 'lon': lon}, 'forecast': result})
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/property-report', methods=['POST'])
def property_report():
    """Return property-level risk features for a point."""
    if dataset is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    try:
        data = request.get_json() or {}
        lat = float(data.get('latitude'))
        lon = float(data.get('longitude'))
        dataset['distance_to_query'] = np.sqrt((dataset['latitude'] - lat)**2 + (dataset['longitude'] - lon)**2)
        nearest_idx = dataset['distance_to_query'].idxmin()
        pt = dataset.iloc[nearest_idx]
        raw = pd.DataFrame([{ 
            'elevation_m': float(pt['elevation_m']),
            'slope_percent': float(pt['slope_percent']),
            'distance_to_water_m': float(pt['distance_to_water_m']),
            'precipitation_mm_event': float(pt['precipitation_mm_event']),
            'soil_permeability_mm_hr': float(pt['soil_permeability_mm_hr']),
            'land_cover': float(pt['land_cover'])
        }])
        p = rule_based_probabilities(raw, 0.0, local_elev_median=float(pt['elevation_m']))
        dataset.drop('distance_to_query', axis=1, inplace=True)
        return jsonify({
            'latitude': lat, 'longitude': lon,
            'elevation_m': float(pt['elevation_m']),
            'distance_to_water_m': float(pt['distance_to_water_m']),
            'soil_permeability_mm_hr': float(pt['soil_permeability_mm_hr']),
            'land_cover': float(pt['land_cover']),
            'base_probability': float(p[0])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reverse-geocode', methods=['POST'])
def reverse_geocode():
    """Reverse geocode lat/lon to a human-readable location label.

    Body: { latitude, longitude }
    Returns: { label, components }
    """
    try:
        data = request.get_json() or {}
        lat = float(data.get('latitude'))
        lon = float(data.get('longitude'))

        url = 'https://nominatim.openstreetmap.org/reverse'
        params = {
            'format': 'jsonv2',
            'lat': lat,
            'lon': lon,
            'zoom': 10,
            'addressdetails': 1
        }
        headers = {'User-Agent': 'FloodScope/1.0'}
        r = requests.get(url, params=params, headers=headers, timeout=20)
        if r.status_code != 200:
            return jsonify({'label': 'Unknown location', 'components': {}})
        j = r.json()
        addr = j.get('address', {})
        city = addr.get('city') or addr.get('town') or addr.get('village') or addr.get('hamlet')
        state = addr.get('state') or addr.get('region') or addr.get('county')
        country = addr.get('country')
        parts = [p for p in [city, state, country] if p]
        label = ', '.join(parts) if parts else 'Unknown location'
        return jsonify({'label': label, 'components': addr})
    except Exception as e:
        return jsonify({'label': 'Unknown location', 'error': str(e)}), 200


@app.route('/api/predict-grid', methods=['POST'])
def predict_grid():
    """Generate dense grid predictions for any location - TRUE GENERALIZATION.
    
    Creates a grid of points within radius, estimates features from nearest neighbors,
    and predicts flood risk for EVERY point - not just training data.
    
    Expected JSON body:
    {
        "latitude": 29.76,
        "longitude": -95.37,
        "radius_meters": 8047,       # default ~5 miles
        "precip_inches": 0.0,        # optional extra precipitation
        "grid_spacing_meters": 200   # spacing between grid points (default 200m)
    }
    """
    if dataset is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    
    try:
        data = request.get_json() or {}
        center_lat = float(data.get('latitude', 29.7604))
        center_lon = float(data.get('longitude', -95.3698))
        radius_m = float(data.get('radius_meters', 8047))
        precip_in = float(data.get('precip_inches', 0.0))
        grid_spacing = float(data.get('grid_spacing_meters', 200))
        max_points = int(data.get('max_points', 4000))
        mode = str(data.get('mode', 'algo')).lower()  # 'algo' or 'ml'
        use_osm_water = bool(data.get('use_osm_water', True))
        
        # Convert radius to degrees (approximate)
        lat_rad = np.radians(center_lat)
        meters_per_deg_lat = 111000
        meters_per_deg_lon = 111000 * np.cos(lat_rad)
        
        radius_deg_lat = radius_m / meters_per_deg_lat
        radius_deg_lon = radius_m / meters_per_deg_lon
        spacing_deg_lat = grid_spacing / meters_per_deg_lat
        spacing_deg_lon = grid_spacing / meters_per_deg_lon
        # BBox for local context
        south = center_lat - radius_deg_lat
        north = center_lat + radius_deg_lat
        west = center_lon - radius_deg_lon
        east = center_lon + radius_deg_lon
        
        # Generate grid points (vectorized)
        lats = np.arange(center_lat - radius_deg_lat, center_lat + radius_deg_lat, spacing_deg_lat)
        lons = np.arange(center_lon - radius_deg_lon, center_lon + radius_deg_lon, spacing_deg_lon)
        grid_lat, grid_lon = np.meshgrid(lats, lons, indexing='ij')
        grid_lat = grid_lat.ravel()
        grid_lon = grid_lon.ravel()

        # Haversine radius mask
        dlat = np.radians(grid_lat - center_lat)
        dlon = np.radians(grid_lon - center_lon)
        a = np.sin(dlat/2.0)**2 + np.cos(np.radians(center_lat)) * np.cos(np.radians(grid_lat)) * np.sin(dlon/2.0)**2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        dist = 6371000.0 * c
        mask = dist <= radius_m
        grid_lat = grid_lat[mask]
        grid_lon = grid_lon[mask]

        # Subsample if too many points to keep response fast
        if grid_lat.size > max_points:
            step = int(np.ceil(grid_lat.size / max_points))
            grid_lat = grid_lat[::step]
            grid_lon = grid_lon[::step]

        if grid_lat.size == 0:
            return jsonify({'type': 'FeatureCollection', 'features': [], 'count': 0})
        
        # KNN-weighted features for grid points (k=8)
        data_rad = np.radians(dataset[['latitude', 'longitude']].values)
        tree = BallTree(data_rad, metric='haversine')
        grid_rad = np.radians(np.c_[grid_lat, grid_lon])
        d_r, nn_idx = tree.query(grid_rad, k=8, return_distance=True)
        w = 1.0 / (d_r + 1e-6)
        w /= w.sum(axis=1, keepdims=True)
        cols = ['elevation_m','slope_percent','distance_to_water_m','precipitation_mm_event','soil_permeability_mm_hr','land_cover']
        vals = dataset[cols].astype(float).to_numpy()
        gathered = vals[nn_idx]
        weighted = (gathered * w[..., None]).sum(axis=1)
        raw_features = pd.DataFrame(weighted, columns=cols)

        # Optionally refine distance_to_water using OSM water features
        if use_osm_water:
            try:
                water_query = f"""
                [out:json][timeout:25];
                (
                  way["waterway"~"river|stream|canal"]({south},{west},{north},{east});
                  way["natural"="water"]({south},{west},{north},{east});
                  relation["natural"="water"]({south},{west},{north},{east});
                  way["landuse"="reservoir"]({south},{west},{north},{east});
                );
                out geom;
                """
                # Try multiple Overpass hosts for reliability
                overpass_hosts = [
                    'https://overpass-api.de/api/interpreter',
                    'https://overpass.kumi.systems/api/interpreter',
                    'https://overpass.openstreetmap.ru/api/interpreter'
                ]
                wr = None
                for host in overpass_hosts:
                    try:
                        wr = requests.post(host, data={'data': water_query}, timeout=30)
                        if wr.status_code == 200: break
                    except Exception:
                        wr = None
                        continue
                water_pts = []
                if wr is not None and wr.status_code == 200:
                    wj = wr.json()
                    for el in wj.get('elements', []):
                        geom = el.get('geometry') or []
                        for g in geom:
                            la = g.get('lat'); lo = g.get('lon')
                            if la is not None and lo is not None:
                                water_pts.append((float(la), float(lo)))
                if len(water_pts) >= 2:
                    water_tree = BallTree(np.radians(np.array(water_pts)), metric='haversine')
                    grid_pts = np.c_[grid_lat, grid_lon]
                    d_rad, _ = water_tree.query(np.radians(grid_pts), k=1)
                    d_m = (d_rad.ravel() * 6371000.0)
                    raw_features['distance_to_water_m'] = d_m
            except Exception:
                pass
        
        if mode == 'ml':
            # ML mode: use slider precipitation as the event precipitation (absolute)
            raw_features['precipitation_mm_event'] = (precip_in * 25.4)
            X_grid = build_feature_matrix(raw_features)
            ml_probs = predict_probabilities_with_names(X_grid)
            # Blend with hydrology algorithm to restore local variability
            bbox = dataset[
                (dataset['latitude'] >= south) & (dataset['latitude'] <= north) &
                (dataset['longitude'] >= west) & (dataset['longitude'] <= east)
            ]
            local_median_elev = float(bbox['elevation_m'].median()) if len(bbox) else float(dataset['elevation_m'].median())
            algo_probs = rule_based_probabilities(raw_features, precip_in, local_elev_median=local_median_elev)
            # Compute per-point alpha based on nearest training distance (smaller -> trust ML more)
            # Reuse previous KNN distances (d_r from tree.query above)
            dmin_m = (d_r.min(axis=1) * 6371000.0)
            alpha = np.exp(-dmin_m / 5000.0)  # within ~5 km -> alpha ~ 0.37..1
            alpha = np.clip(alpha, 0.15, 0.95)
            probs = alpha * ml_probs + (1.0 - alpha) * algo_probs
        else:
            # Algorithm mode: use only slider precip; get local median elevation for context
            # Compute local elevation median once using a bbox
            lat_rad = np.radians(center_lat)
            dlat = radius_m / 111000.0
            dlon = radius_m / (111000.0 * max(np.cos(lat_rad), 1e-6))
            bbox = dataset[
                (dataset['latitude'] >= center_lat - dlat) & (dataset['latitude'] <= center_lat + dlat) &
                (dataset['longitude'] >= center_lon - dlon) & (dataset['longitude'] <= center_lon + dlon)
            ]
            local_median_elev = float(bbox['elevation_m'].median()) if len(bbox) else float(dataset['elevation_m'].median())

            # Optional: refine distance_to_water using OSM water features inside bbox
            if use_osm_water:
                south, north = center_lat - dlat, center_lat + dlat
                west, east = center_lon - dlon, center_lon + dlon
                try:
                    overpass_url = 'https://overpass-api.de/api/interpreter'
                    ov_query = f"""
                    [out:json][timeout:25];
                    (
                      way["waterway"~"river|stream|canal"]({south},{west},{north},{east});
                      way["natural"="water"]({south},{west},{north},{east});
                      relation["natural"="water"]({south},{west},{north},{east});
                      way["landuse"="reservoir"]({south},{west},{north},{east});
                    );
                    out geom;
                    """
                    rsp = requests.post(overpass_url, data={'data': ov_query}, timeout=30)
                    water_pts = []
                    if rsp.status_code == 200:
                        j = rsp.json()
                        for el in j.get('elements', []):
                            geom = el.get('geometry') or []
                            for g in geom:
                                la = g.get('lat'); lo = g.get('lon')
                                if la is not None and lo is not None:
                                    water_pts.append((float(la), float(lo)))
                    if len(water_pts) >= 2:
                        # NN distance to closest water vertex (approx. distance to shoreline/river)
                        water_tree = BallTree(np.radians(np.array(water_pts)), metric='haversine')
                        d_rad, _ = water_tree.query(np.radians(np.c_[grid_lat, grid_lon]), k=1)
                        d_m = (d_rad.ravel() * 6371000.0)
                        raw_features['distance_to_water_m'] = d_m
                except Exception:
                    pass

            probs = rule_based_probabilities(raw_features, precip_in, local_elev_median=local_median_elev)

        # Build GeoJSON features (include elevation and water distance for UI popups)
        features = []
        elev_arr = raw_features['elevation_m'].to_numpy()
        distw_arr = raw_features['distance_to_water_m'].to_numpy()
        for i, (lat, lon, p) in enumerate(zip(grid_lat, grid_lon, probs)):
            p = float(p)
            if p < 0.2:
                risk_level = 'Low'; color = '#10b981'
            elif p < 0.4:
                risk_level = 'Moderate'; color = '#f59e0b'
            elif p < 0.6:
                risk_level = 'High'; color = '#f97316'
            else:
                risk_level = 'Extreme'; color = '#ef4444'

            features.append({
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [float(lon), float(lat)]},
                'properties': {
                    'flood_probability': p,
                    'risk_level': risk_level,
                    'color': color,
                    'grid_point': True,
                    'elevation_m': float(elev_arr[i]) if i < len(elev_arr) else None,
                    'distance_to_water_m': float(distw_arr[i]) if i < len(distw_arr) else None
                }
            })

        return jsonify({'type': 'FeatureCollection', 'features': features, 'count': len(features), 'grid_spacing_m': grid_spacing})
        
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/predict-area', methods=['POST'])
def predict_area():
    """Predict flood risk for all dataset points within a radius of a center.

    Expected JSON body:
    {
        "latitude": 29.76,
        "longitude": -95.37,
        "radius_meters": 8047,       # default ~5 miles
        "precip_inches": 0.0         # optional extra precipitation to add
    }
    Returns GeoJSON FeatureCollection for the subset.
    """
    if dataset is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    try:
        data = request.get_json() or {}
        lat = float(data.get('latitude', 29.7604))
        lon = float(data.get('longitude', -95.3698))
        radius_m = float(data.get('radius_meters', 8047))  # ~5 miles
        precip_in = float(data.get('precip_inches', 0.0))

        # Quick bounding box to reduce candidate points
        lat_rad = np.radians(lat)
        dlat = radius_m / 111000.0
        dlon = radius_m / (111000.0 * max(np.cos(lat_rad), 1e-6))
        bbox = dataset[
            (dataset['latitude'] >= lat - dlat) & (dataset['latitude'] <= lat + dlat) &
            (dataset['longitude'] >= lon - dlon) & (dataset['longitude'] <= lon + dlon)
        ].copy()

        if len(bbox) == 0:
            return jsonify({'type': 'FeatureCollection', 'features': [], 'count': 0})

        # Precise haversine distance filter (meters)
        R = 6371000.0
        lat1 = np.radians(bbox['latitude'].values)
        lon1 = np.radians(bbox['longitude'].values)
        lat0 = np.radians(lat)
        lon0 = np.radians(lon)
        dlat_v = lat1 - lat0
        dlon_v = lon1 - lon0
        a = np.sin(dlat_v/2.0)**2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlon_v/2.0)**2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        dist_m = R * c
        bbox = bbox.loc[dist_m <= radius_m].copy()

        if len(bbox) == 0:
            return jsonify({'type': 'FeatureCollection', 'features': [], 'count': 0})

        # Optional precipitation override (inches -> mm)
        if precip_in and precip_in != 0.0:
            bbox['precipitation_mm_event'] = bbox['precipitation_mm_event'].astype(float) + (precip_in * 25.4)

        # Build features and predict (vectorized and name-safe)
        X = build_feature_matrix(bbox)
        probs = predict_probabilities_with_names(X)

        # Build GeoJSON
        features = []
        for idx, (i, row) in enumerate(bbox.iterrows()):
            p = float(probs[idx])
            if p < 0.2:
                risk_level = 'Low'; color = '#10b981'
            elif p < 0.4:
                risk_level = 'Moderate'; color = '#f59e0b'
            elif p < 0.6:
                risk_level = 'High'; color = '#f97316'
            else:
                risk_level = 'Extreme'; color = '#ef4444'
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [row['longitude'], row['latitude']]
                },
                'properties': {
                    'house_id': str(row.get('house_id', i)),
                    'flood_probability': p,
                    'risk_level': risk_level,
                    'color': color,
                    'elevation_m': float(row.get('elevation_m', 0.0)),
                    'distance_to_water_m': float(row.get('distance_to_water_m', 0.0)),
                    'historical_flood': int(row.get('historical_flood', 0))
                }
            })

        return jsonify({'type': 'FeatureCollection', 'features': features, 'count': len(features)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/live-risk', methods=['POST'])
def calculate_live_risk():
    """
    Calculate live flood risk for a specific location with current weather.
    
    Expected JSON body:
    {
        "latitude": 29.7604,
        "longitude": -95.3698
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'latitude' not in data or 'longitude' not in data:
            return jsonify({'error': 'Missing latitude or longitude'}), 400
        
        lat = float(data['latitude'])
        lon = float(data['longitude'])
        
        # Find nearest point in dataset (for base features)
        dataset['distance_to_query'] = np.sqrt(
            (dataset['latitude'] - lat)**2 + 
            (dataset['longitude'] - lon)**2
        )
        nearest_idx = dataset['distance_to_query'].idxmin()
        nearest_point = dataset.iloc[nearest_idx]
        
        # Prepare base raw features then engineer to model features
        raw_base = pd.DataFrame([{ 
            'elevation_m': float(nearest_point['elevation_m']),
            'slope_percent': float(nearest_point['slope_percent']),
            'distance_to_water_m': float(nearest_point['distance_to_water_m']),
            'precipitation_mm_event': float(nearest_point['precipitation_mm_event']),
            'soil_permeability_mm_hr': float(nearest_point['soil_permeability_mm_hr']),
            'land_cover': float(nearest_point['land_cover'])
        }])
        base_features = build_feature_matrix(raw_base)
        
        # Fetch current weather
        weather_data = get_current_weather(lat, lon)
        
        # Calculate dynamic risk
        risk_assessment = calculate_dynamic_risk(base_features, weather_data)
        
        # Prepare response
        response = {
            'location': {
                'latitude': lat,
                'longitude': lon,
                'nearest_data_point_distance_m': float(nearest_point['distance_to_query'] * 111000)  # Approx conversion to meters
            },
            'risk': risk_assessment,
            'weather': weather_data,
            'base_features': {
                'elevation_m': float(nearest_point['elevation_m']),
                'distance_to_water_m': float(nearest_point['distance_to_water_m']),
                'land_cover_type': nearest_point.get('land_cover_type', 'Unknown')
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Clean up temporary column
        dataset.drop('distance_to_query', axis=1, inplace=True)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def get_statistics():
    """Get summary statistics about the dataset and predictions."""
    if dataset is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    
    try:
        # Prepare features from dataset
        X = build_feature_matrix(dataset)
        probabilities = predict_probabilities_with_names(X)
        
        # Calculate statistics
        at_risk_mask = probabilities >= 0.5
        homes_at_risk = int(np.sum(at_risk_mask))
        stats = {
            'total_locations': len(dataset),
            'risk_distribution': {
                'low': int(np.sum(probabilities < 0.2)),
                'moderate': int(np.sum((probabilities >= 0.2) & (probabilities < 0.4))),
                'high': int(np.sum((probabilities >= 0.4) & (probabilities < 0.6))),
                'extreme': int(np.sum(probabilities >= 0.6))
            },
            'homes_at_risk': homes_at_risk,
            'homes_at_risk_percent': float(homes_at_risk / len(dataset) * 100.0),
            'average_flood_probability': float(np.mean(probabilities)),
            'max_flood_probability': float(np.max(probabilities)),
            'min_flood_probability': float(np.min(probabilities)),
            'historical_floods': int(dataset['historical_flood'].sum())
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ----------------------------------------------------------------------------
# AUXILIARY ENDPOINTS: buildings count (Overpass) and elevation grid for 3D
# ----------------------------------------------------------------------------

@app.route('/api/building-count', methods=['POST'])
def building_count():
    """Estimate number of buildings/houses in the given radius using Overpass API.

    Body: { latitude, longitude, radius_meters }
    Returns: { total_buildings, house_like }
    """
    try:
        data = request.get_json() or {}
        lat = float(data.get('latitude'))
        lon = float(data.get('longitude'))
        radius_m = float(data.get('radius_meters', 8047))

        # Approximate bbox from radius
        lat_rad = np.radians(lat)
        dlat = radius_m / 111000.0
        dlon = radius_m / (111000.0 * max(np.cos(lat_rad), 1e-6))
        south, north = lat - dlat, lat + dlat
        west, east = lon - dlon, lon + dlon

        # Query buildings in bbox
        query = f"""
        [out:json][timeout:25];
        (
          way["building"]({south},{west},{north},{east});
          relation["building"]({south},{west},{north},{east});
        );
        out tags;
        """
        overpass_hosts = [
            'https://overpass-api.de/api/interpreter',
            'https://overpass.kumi.systems/api/interpreter',
            'https://overpass.openstreetmap.ru/api/interpreter'
        ]
        r = None
        for host in overpass_hosts:
            try:
                r = requests.post(host, data={'data': query}, timeout=30)
                if r.status_code == 200: break
            except Exception:
                r = None
                continue
        total = 0
        house_like = 0
        if r is not None and r.status_code == 200:
            j = r.json()
            for el in j.get('elements', []):
                tags = el.get('tags', {}) or {}
                b = tags.get('building')
                if b:
                    total += 1
                    b_norm = str(b).lower()
                    if b_norm in {
                        'house','residential','detached','apartments','terrace','semi','semidetached_house',
                        'farm','bungalow','hut','dormitory','static_caravan','residence','yes'
                    }:
                        house_like += 1
        bbox = {
            'south': float(south), 'north': float(north),
            'west': float(west), 'east': float(east)
        }
        return jsonify({'total_buildings': total, 'house_like': house_like, 'bbox': bbox})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/predict-buildings', methods=['POST'])
def predict_buildings():
    """Return per-building risk predictions using OSM building centroids.

    Body: { latitude, longitude, radius_meters, precip_inches, mode, max_buildings }
    """
    if dataset is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    try:
        data = request.get_json() or {}
        center_lat = float(data.get('latitude'))
        center_lon = float(data.get('longitude'))
        radius_m = float(data.get('radius_meters', 3000))
        precip_in = float(data.get('precip_inches', 0.0))
        mode = str(data.get('mode', 'algo')).lower()
        max_buildings = int(data.get('max_buildings', 8000))
        use_osm_water = bool(data.get('use_osm_water', True))

        # BBox
        lat_rad = np.radians(center_lat)
        dlat = radius_m / 111000.0
        dlon = radius_m / (111000.0 * max(np.cos(lat_rad), 1e-6))
        south, north = center_lat - dlat, center_lat + dlat
        west, east = center_lon - dlon, center_lon + dlon

        # Overpass: building centers (ways/relations) and building nodes + addr nodes
        ov_query = f"""
        [out:json][timeout:30];
        (
          way["building"]({south},{west},{north},{east});
          relation["building"]({south},{west},{north},{east});
          node["building"]({south},{west},{north},{east});
          node["addr:housenumber"]({south},{west},{north},{east});
        );
        out center;
        """
        overpass_hosts = [
            'https://overpass-api.de/api/interpreter',
            'https://overpass.kumi.systems/api/interpreter',
            'https://overpass.openstreetmap.ru/api/interpreter'
        ]
        rsp = None
        for host in overpass_hosts:
            try:
                rsp = requests.post(host, data={'data': ov_query}, timeout=45)
                if rsp.status_code == 200:
                    break
            except Exception:
                rsp = None
                continue
        if rsp is None or rsp.status_code != 200:
            return jsonify({'type': 'FeatureCollection', 'features': [], 'count': 0, 'error': 'Overpass error'}), 200
        centers = []
        bids = []
        for el in (rsp.json().get('elements', []) or []):
            if len(centers) >= max_buildings:
                break
            t = el.get('type')
            if t in ('way', 'relation'):
                cen = el.get('center')
                if cen and 'lat' in cen and 'lon' in cen:
                    centers.append((float(cen['lat']), float(cen['lon'])))
                    bids.append(str(el.get('id')))
            elif t == 'node':
                la = el.get('lat'); lo = el.get('lon')
                if la is not None and lo is not None:
                    centers.append((float(la), float(lo)))
                    bids.append(str(el.get('id')))

        if not centers:
            return jsonify({'type': 'FeatureCollection', 'features': [], 'count': 0})

        # KNN-weighted features from dataset
        pts = np.array(centers)
        data_rad = np.radians(dataset[['latitude', 'longitude']].values)
        tree = BallTree(data_rad, metric='haversine')
        d_r, nn_idx = tree.query(np.radians(pts), k=8, return_distance=True)
        # weights ~ 1/d, avoid inf; add small floor to stabilize
        w = 1.0 / (d_r + 1e-6)
        w /= w.sum(axis=1, keepdims=True)
        cols = ['elevation_m','slope_percent','distance_to_water_m','precipitation_mm_event','soil_permeability_mm_hr','land_cover']
        vals = dataset[cols].astype(float).to_numpy()
        # gather and weight
        gathered = vals[nn_idx]
        weighted = (gathered * w[..., None]).sum(axis=1)
        raw_features = pd.DataFrame(weighted, columns=cols)

        # Predict
        if mode == 'ml':
            # In ML mode, treat slider precip as the event precipitation (absolute),
            # so 0 inches yields near-zero probabilities.
            raw_features['precipitation_mm_event'] = (precip_in * 25.4)
            Xb = build_feature_matrix(raw_features)
            ml_probs = predict_probabilities_with_names(Xb)
            # Blend with hydrology algorithm for variability
            algo_probs = rule_based_probabilities(raw_features, precip_in, local_elev_median=local_median_elev)
            # Use distances from the building KNN (d_r) computed above
            dmin_m = (d_r.min(axis=1) * 6371000.0)
            alpha = np.exp(-dmin_m / 5000.0)
            alpha = np.clip(alpha, 0.15, 0.95)
            probs = alpha * ml_probs + (1.0 - alpha) * algo_probs
        else:
            bbox = dataset[
                (dataset['latitude'] >= south) & (dataset['latitude'] <= north) &
                (dataset['longitude'] >= west) & (dataset['longitude'] <= east)
            ]
            local_median_elev = float(bbox['elevation_m'].median()) if len(bbox) else float(dataset['elevation_m'].median())
            probs = rule_based_probabilities(raw_features, precip_in, local_elev_median=local_median_elev)

        elev_arr = raw_features['elevation_m'].to_numpy()
        distw_arr = raw_features['distance_to_water_m'].to_numpy()
        soilp_arr = raw_features['soil_permeability_mm_hr'].to_numpy()
        if 'historical_flood' in dataset.columns:
            hist_vals = dataset['historical_flood'].astype(float).to_numpy()
            hist_gathered = hist_vals[nn_idx]
            hist_weighted = (hist_gathered * w).sum(axis=1)
            hist_arr = (hist_weighted >= 0.5).astype(int)
        else:
            hist_arr = np.zeros(len(centers), dtype=int)
        def hsg(val):
            if val <= 5: return 'D'
            if val <= 10: return 'C'
            if val <= 20: return 'B'
            return 'A'

        features = []
        for i, ((lat, lon), p) in enumerate(zip(centers, probs)):
            p = float(p)
            if p < 0.2:
                risk_level = 'Low'; color = '#10b981'
            elif p < 0.4:
                risk_level = 'Moderate'; color = '#f59e0b'
            elif p < 0.6:
                risk_level = 'High'; color = '#f97316'
            else:
                risk_level = 'Extreme'; color = '#ef4444'
            features.append({
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [float(lon), float(lat)]},
                'properties': {
                    'building_id': bids[i] if i < len(bids) else None,
                    'flood_probability': p,
                    'risk_level': risk_level,
                    'color': color,
                    'elevation_m': float(elev_arr[i]) if i < len(elev_arr) else None,
                    'distance_to_water_m': float(distw_arr[i]) if i < len(distw_arr) else None,
                    'soil_permeability_mm_hr': float(soilp_arr[i]) if i < len(soilp_arr) else None,
                    'hsg': hsg(float(soilp_arr[i])) if i < len(soilp_arr) else None,
                    'historical_flood': int(hist_arr[i]) if i < len(hist_arr) else 0
                }
            })

        return jsonify({'type': 'FeatureCollection', 'features': features, 'count': len(features)})
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500
@app.route('/api/elevation-grid', methods=['POST'])
def elevation_grid():
    """Return a simple elevation grid derived from nearest dataset elevations.

    Body: { latitude, longitude, radius_meters, grid: 50 }
    Returns: { rows, cols, lats, lons, z } where z is 2D list rows x cols.
    """
    try:
        if dataset is None:
            return jsonify({'error': 'Dataset not loaded'}), 500

        data = request.get_json() or {}
        lat = float(data.get('latitude', 29.7604))
        lon = float(data.get('longitude', -95.3698))
        radius_m = float(data.get('radius_meters', 5000))
        n = int(data.get('grid', 50))

        lat_rad = np.radians(lat)
        meters_per_deg_lat = 111000.0
        meters_per_deg_lon = 111000.0 * np.cos(lat_rad)
        dlat = radius_m / meters_per_deg_lat
        dlon = radius_m / meters_per_deg_lon

        lat_vals = np.linspace(lat - dlat, lat + dlat, n)
        lon_vals = np.linspace(lon - dlon, lon + dlon, n)
        grid_lat, grid_lon = np.meshgrid(lat_vals, lon_vals, indexing='ij')

        # NN elevation via BallTree
        tree = BallTree(np.radians(dataset[['latitude','longitude']].values), metric='haversine')
        nn_idx = tree.query(np.radians(np.c_[grid_lat.ravel(), grid_lon.ravel()]), k=1, return_distance=False).ravel()
        elev_nn = dataset.iloc[nn_idx]['elevation_m'].astype(float).to_numpy()
        Z = elev_nn.reshape((n, n))

        return jsonify({
            'rows': int(n),
            'cols': int(n),
            'lats': lat_vals.tolist(),
            'lons': lon_vals.tolist(),
            'z': Z.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# APPLICATION STARTUP
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("FLOOD SENTINEL - BACKEND SERVER")
    print("=" * 70)
    
    # Load model and data
    if load_model_and_data():
        print("\n✓ Server initialization successful")
        print("\nStarting Flask development server...")
        print("Access the application at: http://localhost:5000")
        print("\nAPI Endpoints:")
        print("  - GET  /                      - Main application")
        print("  - GET  /api/health            - Health check")
        print("  - GET  /api/risk-predictions  - All predictions (GeoJSON)")
        print("  - POST /api/live-risk         - Live risk assessment")
        print("  - GET  /api/stats             - Summary statistics")
        print("\nPress CTRL+C to stop the server")
        print("=" * 70)
        
        # Run Flask development server
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n✗ Failed to load model/data. Cannot start server.")
        print("Please ensure Phase 1 scripts have been run successfully.")
