# Predictive Maintenance Dashboard - Shutdown Script
# This PowerShell script stops all running services
# Phase 3.7.1 - Complete Development Stack

Write-Host "========================================" -ForegroundColor Red
Write-Host " Stopping Predictive Maintenance Dashboard" -ForegroundColor Red
Write-Host "========================================" -ForegroundColor Red
Write-Host ""

Write-Host "Closing all services..." -ForegroundColor Yellow
Write-Host ""

# Kill all node processes (Frontend)
Write-Host "  [1/3] Stopping Frontend (Node.js/Vite)..." -ForegroundColor White
$nodeProcesses = Get-Process -Name "node" -ErrorAction SilentlyContinue
if ($nodeProcesses) {
    Stop-Process -Name "node" -Force -ErrorAction SilentlyContinue
    Write-Host "        ✅ Frontend stopped" -ForegroundColor Green
} else {
    Write-Host "        ℹ️  No frontend process running" -ForegroundColor Gray
}

# Kill all Python processes (Backend, Celery, Flower)
Write-Host "  [2/3] Stopping Backend & Celery (Python)..." -ForegroundColor White
$pythonProcesses = Get-Process -Name "python" -ErrorAction SilentlyContinue
if ($pythonProcesses) {
    Stop-Process -Name "python" -Force -ErrorAction SilentlyContinue
    Write-Host "        ✅ Backend, Celery, and Flower stopped" -ForegroundColor Green
} else {
    Write-Host "        ℹ️  No Python processes running" -ForegroundColor Gray
}

# Additional cleanup for specific process names
Write-Host "  [3/3] Cleanup..." -ForegroundColor White
Stop-Process -Name "uvicorn" -Force -ErrorAction SilentlyContinue
Stop-Process -Name "celery" -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " All services stopped successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "ℹ️  Note: PostgreSQL and Redis services are still running" -ForegroundColor Cyan
Write-Host "   (They run as Windows services and don't need to be stopped)" -ForegroundColor Gray
Write-Host ""
