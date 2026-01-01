@echo off
echo ========================================
echo  Stopping Predictive Maintenance Dashboard
echo ========================================
echo.

echo Closing all PowerShell windows (Backend, Celery, Frontend)...

REM Stop Printer Logger (best-effort)
for /f "tokens=2 delims==" %%p in ('wmic process where "CommandLine like '%%datalogger.py%%'" get ProcessId /value ^| find "ProcessId"') do taskkill /F /PID %%p /T >nul 2>&1

REM Kill all node processes (Frontend)
taskkill /F /IM node.exe /T >nul 2>&1

REM Kill all Python processes (Backend & Celery)
taskkill /F /IM python.exe /T >nul 2>&1

REM Kill uvicorn processes
taskkill /F /IM uvicorn.exe /T >nul 2>&1

REM Kill celery processes
taskkill /F /IM celery.exe /T >nul 2>&1

echo.
echo ========================================
echo  All services stopped successfully!
echo ========================================
echo.
pause
