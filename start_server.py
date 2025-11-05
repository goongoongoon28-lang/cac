"""
Startup wrapper for Flask app with explicit output
"""
import sys
import os

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

print("=" * 70, flush=True)
print("STARTING FLOOD SENTINEL SERVER", flush=True)
print("=" * 70, flush=True)
print("", flush=True)

# Import and run the Flask app
try:
    print("Importing Flask app...", flush=True)
    from app import app, load_model_and_data
    
    print("Loading model and data...", flush=True)
    if load_model_and_data():
        print("✓ Model and data loaded successfully", flush=True)
        print("", flush=True)
        print("Starting Flask server on http://localhost:5000", flush=True)
        print("Press CTRL+C to stop", flush=True)
        print("=" * 70, flush=True)
        print("", flush=True)
        
        # Run Flask
        app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
    else:
        print("✗ Failed to load model/data", flush=True)
        sys.exit(1)
        
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
