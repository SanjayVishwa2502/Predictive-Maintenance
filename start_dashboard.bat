@echo off
echo ========================================
echo  Starting Predictive Maintenance Dashboard
echo ========================================
echo.

REM Start Printer Logger (MQTT -> CSV)
echo [0/3] Starting Printer MQTT Logger...
start "Printer Logger" powershell -NoExit -Command "cd 'C:\Projects\Predictive Maintenance\3Dprinterdata'; & 'C:\Projects\Predictive Maintenance\venv\Scripts\Activate.ps1'; if (-not $env:PM_MACHINE_ID) { $env:PM_MACHINE_ID = 'printer_creality_ender3_001' }; if (-not $env:PM_LOG_MODE) { $env:PM_LOG_MODE = 'clean' }; if (-not $env:PM_LOG_DIR) { $env:PM_LOG_DIR = 'C:\Projects\Predictive Maintenance\3Dprinterdata' }; Write-Host '=== PRINTER LOGGER (MQTT -> CSV) ===' -ForegroundColor Green; & 'C:\Projects\Predictive Maintenance\venv\Scripts\python.exe' 'C:\Projects\Predictive Maintenance\3Dprinterdata\datalogger.py'"

REM Start Backend (FastAPI)
echo [1/3] Starting Backend Server...
start "Backend - FastAPI" powershell -NoExit -Command "cd 'C:\Projects\Predictive Maintenance\frontend\server'; & 'C:\Projects\Predictive Maintenance\venv\Scripts\Activate.ps1'; Write-Host '=== BACKEND SERVER ===' -ForegroundColor Green; uvicorn main:app --host 0.0.0.0 --port 8000"

REM Wait 3 seconds
timeout /t 3 /nobreak >nul

REM Start Celery Worker
echo [2/3] Starting Celery Worker...
REM NOTE: Celery uses Redis as a broker; pending tasks from earlier runs may execute when the worker starts.
REM Set PM_CELERY_PURGE_ON_START=1 to discard pending tasks on startup.
start "Celery Worker" powershell -NoExit -Command "cd 'C:\Projects\Predictive Maintenance\frontend\server'; & 'C:\Projects\Predictive Maintenance\venv\Scripts\Activate.ps1'; Write-Host '=== CELERY WORKER ===' -ForegroundColor Yellow; if ($env:PM_CELERY_PURGE_ON_START -eq '1') { Write-Host '=== CELERY PURGE (discarding pending tasks) ===' -ForegroundColor Yellow; celery -A celery_app purge -Q celery,gan -f }; celery -A celery_app worker --loglevel=info --pool=solo"

REM Wait 3 seconds
timeout /t 3 /nobreak >nul

REM Start Frontend (Vite)
echo [3/3] Starting Frontend Development Server...
start "Frontend - Vite" powershell -NoExit -Command "cd 'C:\Projects\Predictive Maintenance\frontend\client'; Write-Host '=== FRONTEND SERVER ===' -ForegroundColor Cyan; npm run dev"

REM Wait 5 seconds for servers to start
timeout /t 5 /nobreak >nul

echo.
echo ========================================
echo  All services started successfully!
echo ========================================
echo.
echo  Backend:  http://localhost:8000
echo  API Docs: http://localhost:8000/docs
echo  Frontend: http://localhost:5173
echo.
echo  Press any key to open frontend in browser...
pause >nul

REM Open frontend in default browser
start http://localhost:5173

echo.
echo To stop all services, close all PowerShell windows.
echo.
pause
