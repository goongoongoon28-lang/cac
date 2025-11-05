@echo off
REM Flood Sentinel - Enhanced Server Startup
REM Logs all output to server_log.txt for debugging

echo ======================================================================
echo FLOOD SENTINEL - STARTING SERVER
echo ======================================================================
echo.
echo Logs will be written to: server_log.txt
echo.
echo Starting server... (check server_log.txt for details)
echo.

python run_server.py

echo.
echo ======================================================================
echo Check server_log.txt for detailed startup information
echo ======================================================================
pause
