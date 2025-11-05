"""
Flood Sentinel Server Startup Script
Starts Flask server with comprehensive logging to file
"""
import sys
import os
from datetime import datetime
import traceback

# Setup logging to file
LOG_FILE = 'server_log.txt'

def log_message(msg, to_console=True):
    """Write message to both log file and console"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_msg = f"[{timestamp}] {msg}"
    
    # Write to file
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(full_msg + '\n')
        f.flush()
    
    # Also print to console
    if to_console:
        print(full_msg, flush=True)

def main():
    # Clear previous log
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"=== FLOOD SENTINEL SERVER LOG ===\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 70 + "\n\n")
    
    log_message("=" * 70)
    log_message("FLOOD SENTINEL - SERVER STARTUP")
    log_message("=" * 70)
    log_message("")
    
    try:
        log_message("Step 1: Checking Python version...")
        log_message(f"Python version: {sys.version}")
        log_message("")
        
        log_message("Step 2: Checking working directory...")
        log_message(f"Working directory: {os.getcwd()}")
        log_message("")
        
        log_message("Step 3: Verifying required files...")
        required_files = [
            'app.py',
            'models/flood_risk_model.pkl',
            'models/feature_scaler.pkl',
            'data/final_training_dataset.csv'
        ]
        
        for file in required_files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                log_message(f"  ✓ Found {file} ({size:,} bytes)")
            else:
                log_message(f"  ✗ MISSING: {file}")
                raise FileNotFoundError(f"Required file not found: {file}")
        log_message("")
        
        log_message("Step 4: Importing Flask and dependencies...")
        try:
            import flask
            log_message(f"  ✓ Flask version: {flask.__version__}")
        except ImportError as e:
            log_message(f"  ✗ Flask not found: {e}")
            log_message("\n  To install Flask, run:")
            log_message("  pip install -r requirements.txt")
            raise
        
        try:
            import pandas
            log_message(f"  ✓ Pandas version: {pandas.__version__}")
        except ImportError as e:
            log_message(f"  ✗ Pandas not found: {e}")
            raise
            
        try:
            import numpy
            log_message(f"  ✓ NumPy version: {numpy.__version__}")
        except ImportError as e:
            log_message(f"  ✗ NumPy not found: {e}")
            raise
        
        try:
            import sklearn
            log_message(f"  ✓ Scikit-learn version: {sklearn.__version__}")
        except ImportError as e:
            log_message(f"  ✗ Scikit-learn not found: {e}")
            raise
        log_message("")
        
        log_message("Step 5: Loading Flask application...")
        from app import app, load_model_and_data
        log_message("  ✓ Flask app imported successfully")
        log_message("")
        
        log_message("Step 6: Loading model and data...")
        if load_model_and_data():
            log_message("  ✓ Model and data loaded successfully")
        else:
            log_message("  ✗ Failed to load model/data")
            raise Exception("Model loading failed")
        log_message("")
        
        log_message("=" * 70)
        log_message("SERVER STARTED SUCCESSFULLY!")
        log_message("=" * 70)
        log_message("")
        log_message("Access the application at:")
        log_message("  → http://localhost:5000")
        log_message("  → http://127.0.0.1:5000")
        log_message("")
        log_message("Server logs are being written to: server_log.txt")
        log_message("Press CTRL+C to stop the server")
        log_message("=" * 70)
        log_message("")
        
        # Run Flask with minimal output (logs go to file)
        log_message("Starting Flask development server...", to_console=False)
        app.run(
            debug=False,  # Disable debug to reduce console spam
            host='0.0.0.0',
            port=5000,
            use_reloader=False  # Disable reloader to avoid double startup
        )
        
    except KeyboardInterrupt:
        log_message("\n\nServer stopped by user (CTRL+C)")
        log_message("Goodbye!")
        
    except Exception as e:
        log_message("\n\n" + "=" * 70)
        log_message("ERROR DURING STARTUP")
        log_message("=" * 70)
        log_message(f"Error: {str(e)}")
        log_message("\nFull traceback:")
        log_message(traceback.format_exc())
        log_message("=" * 70)
        log_message("\nPlease fix the error above and try again.")
        log_message("Check server_log.txt for complete details.")
        return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
