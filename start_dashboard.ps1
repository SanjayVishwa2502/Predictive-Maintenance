# Predictive Maintenance Dashboard - Optimized Background Startup Script
# This PowerShell script starts all services in hidden windows for background operation
# Optimized for: Faster startup, better process management, comprehensive logging
# Phase 3.7.3 - Production-Ready Stack

# ============================================================================
# CONFIGURATION
# ============================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = "C:\Projects\Predictive Maintenance"
$LogDir = "$ProjectRoot\logs"
$Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"

# Optional: enable uvicorn auto-reload for development.
# Set env var PM_UVICORN_RELOAD=1 to restart the API server on code changes.
$UvicornReload = ($env:PM_UVICORN_RELOAD -eq "1")

# Optional: purge any pending Celery tasks from previous runs.
# Why: Celery uses Redis as a broker; if a training task was queued earlier and the worker
# was stopped, the task can remain pending and will execute as soon as the worker starts.
# Set env var PM_CELERY_PURGE_ON_START=1 to discard pending tasks on startup.
$PurgeCeleryOnStart = ($env:PM_CELERY_PURGE_ON_START -eq "1")

# Create logs directory if it doesn't exist
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

function Write-ColorOutput($Message, $Color = "White") {
    Write-Host $Message -ForegroundColor $Color
}

function Test-Port($Port) {
    $tcpConnection = Test-NetConnection -ComputerName localhost -Port $Port -WarningAction SilentlyContinue
    return $tcpConnection.TcpTestSucceeded
}

function Wait-ForService($ServiceName, $Port, $MaxWaitSeconds = 30) {
    Write-ColorOutput "  [...] Waiting for $ServiceName (port $Port)..." "Gray"
    $waited = 0
    while ($waited -lt $MaxWaitSeconds) {
        if (Test-Port $Port) {
            Write-ColorOutput "  [OK] $ServiceName is ready!" "Green"
            return $true
        }
        Start-Sleep -Seconds 1
        $waited++
    }
    Write-ColorOutput "  [!] $ServiceName didn't respond within ${MaxWaitSeconds}s" "Yellow"
    return $false
}

# ============================================================================
# STARTUP BANNER
# ============================================================================
Clear-Host
Write-ColorOutput "========================================" "Cyan"
Write-ColorOutput " Predictive Maintenance Dashboard" "Cyan"
Write-ColorOutput " Optimized Background Startup" "Cyan"
Write-ColorOutput "========================================" "Cyan"
Write-ColorOutput ""

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================
Write-ColorOutput "[PREFLIGHT] Checking system services..." "Yellow"

# Check PostgreSQL
$postgres = Get-Service -Name "postgresql-x64-18" -ErrorAction SilentlyContinue
if ($null -eq $postgres) {
    Write-ColorOutput "  [!] PostgreSQL service not found (optional)" "Yellow"
} elseif ($postgres.Status -ne "Running") {
    Write-ColorOutput "  [X] PostgreSQL is not running!" "Red"
    Write-ColorOutput "     Starting PostgreSQL..." "Yellow"
    Start-Service postgresql-x64-18
    Start-Sleep -Seconds 2
    Write-ColorOutput "  [OK] PostgreSQL started" "Green"
} else {
    Write-ColorOutput "  [OK] PostgreSQL running" "Green"
}

# Check Redis
$redis = Get-Service -Name "Redis" -ErrorAction SilentlyContinue
if ($null -eq $redis) {
    Write-ColorOutput "  [!] Redis service not found (optional)" "Yellow"
} elseif ($redis.Status -ne "Running") {
    Write-ColorOutput "  [X] Redis is not running!" "Red"
    Write-ColorOutput "     Starting Redis..." "Yellow"
    Start-Service Redis
    Start-Sleep -Seconds 2
    Write-ColorOutput "  [OK] Redis started" "Green"
} else {
    Write-ColorOutput "  [OK] Redis running" "Green"
}

Write-ColorOutput ""

# ============================================================================
# KILL EXISTING PROCESSES (Cleanup)
# ============================================================================
Write-ColorOutput "[CLEANUP] Checking for existing processes..." "Yellow"

$ports = @(8000, 5555, 5173)
$killed = 0
foreach ($port in $ports) {
    $process = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | 
               Select-Object -ExpandProperty OwningProcess -Unique
    if ($process) {
        Stop-Process -Id $process -Force -ErrorAction SilentlyContinue
        $killed++
        Write-ColorOutput "  [-] Killed process on port $port" "Gray"
    }
}

if ($killed -eq 0) {
    Write-ColorOutput "  [OK] No conflicting processes found" "Green"
} else {
    Write-ColorOutput "  [OK] Cleaned up $killed process(es)" "Green"
    Start-Sleep -Seconds 2
}

Write-ColorOutput ""

# ============================================================================
# START SERVICES (Hidden Background Windows)
# ============================================================================
Write-ColorOutput "[STARTUP] Launching services in background..." "Green"

# 1. Backend (FastAPI) - System PowerShell Window
Write-ColorOutput "[1/4] Starting Backend Server (FastAPI)..." "Green"
$backendLog = "$LogDir\backend_$Timestamp.log"
$uvicornArgs = "main:app --host 0.0.0.0 --port 8000"
if ($UvicornReload) {
    $uvicornArgs = "$uvicornArgs --reload"
}
$backendCmd = "cd '$ProjectRoot\frontend\server'; & '$ProjectRoot\venv\Scripts\Activate.ps1'; Write-Host '=== BACKEND SERVER (Port 8000) ===' -ForegroundColor Green; uvicorn $uvicornArgs 2>&1 | Tee-Object -FilePath '$backendLog'"
Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList "-NoExit", "-Command", $backendCmd
Write-ColorOutput "     Log: $backendLog" "Gray"

# 2. Celery Worker - System PowerShell Window
Write-ColorOutput "[2/4] Starting Celery Worker..." "Yellow"
$celeryLog = "$LogDir\celery_$Timestamp.log"
$celeryPurgeCmd = ""
if ($PurgeCeleryOnStart) {
    # Purge common queues. Routes are currently disabled, but we include 'gan' defensively.
    $celeryPurgeCmd = "Write-Host '=== CELERY PURGE (discarding pending tasks) ===' -ForegroundColor Yellow; celery -A celery_app purge -Q celery,gan -f; "
}

$celeryCmd = "cd '$ProjectRoot\frontend\server'; & '$ProjectRoot\venv\Scripts\Activate.ps1'; Write-Host '=== CELERY WORKER ===' -ForegroundColor Yellow; $celeryPurgeCmd celery -A celery_app worker --loglevel=info --pool=solo 2>&1 | Tee-Object -FilePath '$celeryLog'"
Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList "-NoExit", "-Command", $celeryCmd
Write-ColorOutput "     Log: $celeryLog" "Gray"

# 3. Flower (Celery Monitoring) - System PowerShell Window
Write-ColorOutput "[3/4] Starting Flower (Celery Monitoring)..." "Magenta"
$flowerLog = "$LogDir\flower_$Timestamp.log"
$flowerCmd = "cd '$ProjectRoot\frontend\server'; & '$ProjectRoot\venv\Scripts\Activate.ps1'; Write-Host '=== FLOWER MONITORING (Port 5555) ===' -ForegroundColor Magenta; celery -A celery_app flower --port=5555 2>&1 | Tee-Object -FilePath '$flowerLog'"
Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList "-NoExit", "-Command", $flowerCmd
Write-ColorOutput "     Log: $flowerLog" "Gray"

# 4. Frontend (Vite) - System PowerShell Window
Write-ColorOutput "[4/4] Starting Frontend Server (React)..." "Cyan"
$frontendLog = "$LogDir\frontend_$Timestamp.log"
$frontendCmd = "cd '$ProjectRoot\frontend\client'; Write-Host '=== FRONTEND SERVER (Port 5173) ===' -ForegroundColor Cyan; npm run dev 2>&1 | Tee-Object -FilePath '$frontendLog'"
Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList "-NoExit", "-Command", $frontendCmd
Write-ColorOutput "     Log: $frontendLog" "Gray"

Write-ColorOutput ""

# ============================================================================
# HEALTH CHECKS
# ============================================================================
Write-ColorOutput "[HEALTH] Verifying services..." "Cyan"

$services = @(
    @{Name="Backend API"; Port=8000; Wait=15},
    @{Name="Flower"; Port=5555; Wait=10},
    @{Name="Frontend"; Port=5173; Wait=15}
)

$allHealthy = $true
foreach ($service in $services) {
    $healthy = Wait-ForService $service.Name $service.Port $service.Wait
    if (-not $healthy) { $allHealthy = $false }
}

Write-ColorOutput ""

# ============================================================================
# SUMMARY
# ============================================================================
if ($allHealthy) {
    Write-ColorOutput "========================================" "Green"
    Write-ColorOutput " [SUCCESS] All services started!" "Green"
    Write-ColorOutput "========================================" "Green"
} else {
    Write-ColorOutput "========================================" "Yellow"
    Write-ColorOutput " [WARNING] Some services need more time" "Yellow"
    Write-ColorOutput "========================================" "Yellow"
}

Write-ColorOutput ""
Write-ColorOutput "[SERVICE URLS]" "White"
Write-ColorOutput "  Backend API:     http://localhost:8000" "Cyan"
Write-ColorOutput "  API Docs:        http://localhost:8000/docs" "Cyan"
Write-ColorOutput "  Frontend:        http://localhost:5173" "Cyan"
Write-ColorOutput "  Flower Monitor:  http://localhost:5555" "Magenta"

Write-ColorOutput ""
Write-ColorOutput "[LOGS] Saved to: $LogDir" "Gray"

Write-ColorOutput ""
Write-ColorOutput "[COMMANDS] Management:" "Yellow"
Write-ColorOutput "  Stop services:   .\stop_dashboard.ps1" "Gray"
Write-ColorOutput "  View logs:       Get-Content `"$LogDir\*_$Timestamp.log`" -Tail 50 -Wait" "Gray"
Write-ColorOutput "  Check status:    Get-NetTCPConnection -LocalPort 8000,5555,5173" "Gray"

Write-ColorOutput ""
Write-ColorOutput "[BROWSER] Opening frontend..." "Cyan"
Start-Sleep -Seconds 2
Start-Process "http://localhost:5173"

Write-ColorOutput ""
Write-ColorOutput "[SUCCESS] Dashboard is running in background. Close this window safely." "Green"
Write-ColorOutput ""
