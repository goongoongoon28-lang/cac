"""
Quick API Key Test Script
==========================
Tests that your API keys are loaded correctly and working.
"""

import requests
from config import NOAA_API_KEY, OPENWEATHER_API_KEY

print("=" * 70)
print("API KEY VALIDATION TEST")
print("=" * 70)

# Test 1: Check if keys are loaded
print("\n[Test 1] Checking if API keys are loaded from .env file...")
if NOAA_API_KEY:
    print(f"   ✓ NOAA API Key loaded: {NOAA_API_KEY[:8]}...{NOAA_API_KEY[-4:]}")
else:
    print(f"   ✗ NOAA API Key NOT found!")

if OPENWEATHER_API_KEY:
    print(f"   ✓ OpenWeather API Key loaded: {OPENWEATHER_API_KEY[:8]}...{OPENWEATHER_API_KEY[-4:]}")
else:
    print(f"   ✗ OpenWeather API Key NOT found!")

# Test 2: Test NOAA API with a simple request
print("\n[Test 2] Testing NOAA API connection...")
if NOAA_API_KEY:
    try:
        url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/datasets"
        headers = {'token': NOAA_API_KEY}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            print(f"   ✓ NOAA API is working! Status: {response.status_code}")
            data = response.json()
            print(f"   ✓ Available datasets: {data['metadata']['resultset']['count']}")
        else:
            print(f"   ⚠ NOAA API returned status: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   ✗ NOAA API test failed: {e}")
else:
    print("   ⊘ Skipping NOAA test (no key found)")

# Test 3: Test OpenWeather API
print("\n[Test 3] Testing OpenWeather API connection...")
if OPENWEATHER_API_KEY:
    try:
        # Test with Houston coordinates
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': 29.7604,
            'lon': -95.3698,
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric'
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            print(f"   ✓ OpenWeather API is working! Status: {response.status_code}")
            data = response.json()
            print(f"   ✓ Current Houston weather: {data['weather'][0]['description']}")
            print(f"   ✓ Temperature: {data['main']['temp']}°C")
        else:
            print(f"   ⚠ OpenWeather API returned status: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   ✗ OpenWeather API test failed: {e}")
else:
    print("   ⊘ Skipping OpenWeather test (no key found)")

# Test 4: Test USGS 3DEP (no key needed)
print("\n[Test 4] Testing USGS 3DEP Elevation API (no key needed)...")
try:
    url = "https://epqs.nationalmap.gov/v1/json"
    params = {
        'x': -95.3698,  # Houston longitude
        'y': 29.7604,   # Houston latitude
        'units': 'Meters',
        'output': 'json'
    }
    response = requests.get(url, params=params, timeout=10)
    
    if response.status_code == 200:
        print(f"   ✓ USGS 3DEP API is working! Status: {response.status_code}")
        data = response.json()
        if 'value' in data:
            print(f"   ✓ Houston elevation: {data['value']} meters")
    else:
        print(f"   ⚠ USGS 3DEP returned status: {response.status_code}")
except Exception as e:
    print(f"   ✗ USGS 3DEP test failed: {e}")

print("\n" + "=" * 70)
print("API VALIDATION COMPLETE")
print("=" * 70)
print("\nIf all tests passed, you're ready to run the Phase 1 scripts!")
print("Next: Run 'python 1_build_base_dataset.py'")
print("=" * 70)
