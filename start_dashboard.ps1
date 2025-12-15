# Predictive Maintenance Dashboard - Startup Script
# This PowerShell script starts all required services
# Phase 3.7.1 - Complete Development Stack

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Starting Predictive Maintenance Dashboard" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if PostgreSQL and Redis services are running
Write-Host "Checking required services..." -ForegroundColor Gray
$postgres = Get-Service -Name "postgresql-x64-18" -ErrorAction SilentlyContinue
$redis = Get-Service -Name "Redis" -ErrorAction SilentlyContinue

if ($postgres.Status -ne "Running") {
    Write-Host "  ❌ PostgreSQL is not running!" -ForegroundColor Red
    Write-Host "     Run: Start-Service postgresql-x64-18" -ForegroundColor Yellow
    exit 1
}
Write-Host "  ✅ PostgreSQL running" -ForegroundColor Green

if ($redis.Status -ne "Running") {
    Write-Host "  ❌ Redis is not running!" -ForegroundColor Red
    Write-Host "     Run: Start-Service Redis" -ForegroundColor Yellow
    exit 1
}
Write-Host "  ✅ Redis running" -ForegroundColor Green
Write-Host ""

# Start Backend (FastAPI)
Write-Host "[1/4] Starting Backend Server (FastAPI)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Projects\Predictive Maintenance\frontend\server'; & 'C:\Projects\Predictive Maintenance\venv\Scripts\Activate.ps1'; Write-Host '=== BACKEND SERVER (Port 8000) ===' -ForegroundColor Green; uvicorn main:app --reload --host 0.0.0.0 --port 8000"

# Wait 3 seconds
Start-Sleep -Seconds 3

# Start Celery Worker
Write-Host "[2/4] Starting Celery Worker..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Projects\Predictive Maintenance\frontend\server'; & 'C:\Projects\Predictive Maintenance\venv\Scripts\Activate.ps1'; Write-Host '=== CELERY WORKER ===' -ForegroundColor Yellow; celery -A celery_app worker --loglevel=info --pool=solo"

# Wait 3 seconds
Start-Sleep -Seconds 3

# Start Flower (Celery Monitoring)
Write-Host "[3/4] Starting Flower (Celery Monitoring)..." -ForegroundColor Magenta
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Projects\Predictive Maintenance\frontend\server'; & 'C:\Projects\Predictive Maintenance\venv\Scripts\Activate.ps1'; Write-Host '=== FLOWER MONITORING (Port 5555) ===' -ForegroundColor Magenta; celery -A celery_app flower --port=5555"

# Wait 3 seconds
Start-Sleep -Seconds 3

# Start Frontend (Vite)
Write-Host "[4/4] Starting Frontend Development Server (React)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Projects\Predictive Maintenance\frontend\client'; Write-Host '=== FRONTEND SERVER (Port 5173) ===' -ForegroundColor Cyan; npm run dev"

# Wait 5 seconds for servers to start
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " All services started successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Services Running:" -ForegroundColor White
Write-Host ""
Write-Host "  Backend API:     http://localhost:8000" -ForegroundColor White
Write-Host "  API Docs:        http://localhost:8000/docs" -ForegroundColor White
Write-Host "  Frontend:        http://localhost:5173" -ForegroundColor White
Write-Host "  Flower Monitor:  http://localhost:5555" -ForegroundColor White
Write-Host ""
Write-Host "🔧 Background Services:" -ForegroundColor White
Write-Host "  - Celery Worker (processing async tasks)" -ForegroundColor Gray
Write-Host "  - PostgreSQL (database)" -ForegroundColor Gray
Write-Host "  - Redis (message broker)" -ForegroundColor Gray
Write-Host ""
Write-Host "Opening frontend in browser..." -ForegroundColor Cyan

# Wait 2 seconds
Start-Sleep -Seconds 2

# Open frontend in default browser
Start-Process "http://localhost:5173"

Write-Host ""
Write-Host "💡 Tips:" -ForegroundColor Yellow
Write-Host "  - Check task status: http://localhost:5555" -ForegroundColor Gray
Write-Host "  - API documentation: http://localhost:8000/docs" -ForegroundColor Gray
Write-Host "  - To stop all services: .\stop_dashboard.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "✨ Dashboard is ready! Happy coding!" -ForegroundColor Green
Write-Host ""
