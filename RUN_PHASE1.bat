@echo off
REM Flood Sentinel - Phase 1 Execution Script
REM ==========================================
REM This batch file runs all Phase 1 scripts in sequence
REM NOW USING REAL API DATA (USGS + NOAA)!

echo ======================================================================
echo FLOOD SENTINEL - PHASE 1: DATA ENGINEERING AND MODEL TRAINING
echo Using REAL data from USGS and NOAA APIs
echo ======================================================================
echo.

echo [Step 0/3] Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)
echo.

echo [Installing Dependencies]
echo Installing required Python packages...
pip install -r requirements_phase1.txt
echo.

echo ======================================================================
echo [Step 1/3] Building Base Dataset
echo ======================================================================
python 1_build_base_dataset.py
if errorlevel 1 (
    echo ERROR: Script 1 failed
    pause
    exit /b 1
)
echo.

echo ======================================================================
echo [Step 2/3] Enriching Dataset with Environmental Features
echo ======================================================================
python 2_enrich_dataset.py
if errorlevel 1 (
    echo ERROR: Script 2 failed
    pause
    exit /b 1
)
echo.

echo ======================================================================
echo [Step 3/3] Training Machine Learning Model
echo ======================================================================
python 3_train_model.py
if errorlevel 1 (
    echo ERROR: Script 3 failed
    pause
    exit /b 1
)
echo.

echo ======================================================================
echo PHASE 1 COMPLETE!
echo ======================================================================
echo.
echo Generated Files:
echo   - data/base_points.csv
echo   - data/final_training_dataset.csv
echo   - models/flood_risk_model.pkl
echo   - models/feature_scaler.pkl
echo   - results/confusion_matrix.png
echo   - results/feature_importance.png
echo   - results/roc_curve.png
echo   - results/probability_distribution.png
echo.
echo Next: Review the results and await confirmation to proceed to Phase 2
echo ======================================================================
pause
