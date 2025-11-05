@echo off
REM Flood Sentinel - Start Development Server
REM ==========================================
REM No admin access required!

echo ======================================================================
echo FLOOD SENTINEL - STARTING WEB SERVER
echo No admin access required - runs on your user account
echo ======================================================================
echo.

echo Starting Flask server on http://localhost:5000
echo.
echo Available at:
echo   - http://localhost:5000          (local access)
echo   - http://127.0.0.1:5000          (local access)
echo.
echo Press CTRL+C to stop the server
echo ======================================================================
echo.

python app.py

pause
