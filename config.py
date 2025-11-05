"""
Configuration file for API keys and constants
==============================================
Loads API keys from .env file for secure access
"""

import os
from pathlib import Path

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file."""
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Load the .env file
load_env_file()

# API Keys
OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', '')
NOAA_API_KEY = os.environ.get('NOAA_API_KEY', '')

# Validate keys are present
if not OPENWEATHER_API_KEY:
    print("⚠ Warning: OPENWEATHER_API_KEY not found in .env file")

if not NOAA_API_KEY:
    print("⚠ Warning: NOAA_API_KEY not found in .env file")
