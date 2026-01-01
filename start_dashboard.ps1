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

# Stop any existing datalogger instances (no port binding; match command line)
try {
    $loggerProcs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*3Dprinterdata*datalogger.py*" }
    foreach ($p in ($loggerProcs | Select-Object -Unique ProcessId)) {
        try {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
            Write-ColorOutput "  [-] Killed datalogger (PID: $($p.ProcessId))" "Gray"
        } catch {
            # ignore
        }
    }
} catch {
    # ignore
}

Write-ColorOutput ""

# ============================================================================
# START SERVICES (Hidden Background Windows)
# ============================================================================
Write-ColorOutput "[STARTUP] Launching services in background..." "Green"

# Optional: start a simple monitor that logs API/port up/down transitions.
# Set env var PM_MONITOR_ON_START=1 to enable.
$MonitorOnStart = ($env:PM_MONITOR_ON_START -eq "1")

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

# 1b. Printer Logger (MQTT -> clean CSV)
# This is NOT a separate backend; it's a long-running data capture process.
Write-ColorOutput "[LOGGER] Starting Printer MQTT Logger..." "Green"
$loggerLog = "$LogDir\logger_$Timestamp.log"
$loggerCmd = "cd '$ProjectRoot\3Dprinterdata'; & '$ProjectRoot\venv\Scripts\Activate.ps1'; " +
    "if (-not `$env:PM_MACHINE_ID) { `$env:PM_MACHINE_ID = 'printer_creality_ender3_001' }; " +
    "if (-not `$env:PM_LOG_MODE) { `$env:PM_LOG_MODE = 'clean' }; " +
    "if (-not `$env:PM_LOG_DIR) { `$env:PM_LOG_DIR = '$ProjectRoot\3Dprinterdata' }; " +
    "Write-Host '=== PRINTER LOGGER (MQTT -> CSV) ===' -ForegroundColor Green; " +
    "& '$ProjectRoot\venv\Scripts\python.exe' '$ProjectRoot\3Dprinterdata\datalogger.py' 2>&1 | Tee-Object -FilePath '$loggerLog'"
Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList "-NoExit", "-Command", $loggerCmd
Write-ColorOutput "     Log: $loggerLog" "Gray"

# 2. Celery Worker - System PowerShell Window
# Default worker handles GAN/ML and lightweight tasks.
Write-ColorOutput "[2/5] Starting Celery Worker (default)..." "Yellow"
$celeryLog = "$LogDir\celery_$Timestamp.log"
$celeryPurgeCmd = ""
if ($PurgeCeleryOnStart) {
    # Purge common queues. Routes are currently disabled, but we include 'gan' defensively.
    $celeryPurgeCmd = "Write-Host '=== CELERY PURGE (discarding pending tasks) ===' -ForegroundColor Yellow; celery -A celery_app purge -Q celery,gan -f; "
}

$celeryCmd = "cd '$ProjectRoot\frontend\server'; & '$ProjectRoot\venv\Scripts\Activate.ps1'; Write-Host '=== CELERY WORKER (default) ===' -ForegroundColor Yellow; $celeryPurgeCmd celery -A celery_app worker --loglevel=info --pool=solo -Q celery,default,gan,ml 2>&1 | Tee-Object -FilePath '$celeryLog'"
Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList "-NoExit", "-Command", $celeryCmd
Write-ColorOutput "     Log: $celeryLog" "Gray"

# 3. Celery Worker (LLM) - System PowerShell Window
# Dedicated queue for long CPU inference to avoid blocking other tasks.
Write-ColorOutput "[3/5] Starting Celery Worker (LLM queue)..." "Yellow"
$celeryLlmLog = "$LogDir\celery_llm_$Timestamp.log"
$celeryLlmCmd = "cd '$ProjectRoot\frontend\server'; & '$ProjectRoot\venv\Scripts\Activate.ps1'; Write-Host '=== CELERY WORKER (llm) ===' -ForegroundColor Yellow; celery -A celery_app worker --loglevel=info --pool=solo -Q llm 2>&1 | Tee-Object -FilePath '$celeryLlmLog'"
Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList "-NoExit", "-Command", $celeryLlmCmd
Write-ColorOutput "     Log: $celeryLlmLog" "Gray"

# 4. Flower (Celery Monitoring) - System PowerShell Window
Write-ColorOutput "[4/5] Starting Flower (Celery Monitoring)..." "Magenta"
$flowerLog = "$LogDir\flower_$Timestamp.log"
$flowerCmd = "cd '$ProjectRoot\frontend\server'; & '$ProjectRoot\venv\Scripts\Activate.ps1'; Write-Host '=== FLOWER MONITORING (Port 5555) ===' -ForegroundColor Magenta; celery -A celery_app flower --port=5555 2>&1 | Tee-Object -FilePath '$flowerLog'"
Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList "-NoExit", "-Command", $flowerCmd
Write-ColorOutput "     Log: $flowerLog" "Gray"

# 5. Frontend (Vite) - System PowerShell Window
Write-ColorOutput "[5/5] Starting Frontend Server (React)..." "Cyan"
$frontendLog = "$LogDir\frontend_$Timestamp.log"
$frontendCmd = "cd '$ProjectRoot\frontend\client'; Write-Host '=== FRONTEND SERVER (Port 5173) ===' -ForegroundColor Cyan; npm run dev 2>&1 | Tee-Object -FilePath '$frontendLog'"
Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList "-NoExit", "-Command", $frontendCmd
Write-ColorOutput "     Log: $frontendLog" "Gray"

# 6. Monitor (optional)
if ($MonitorOnStart) {
    Write-ColorOutput "[MONITOR] Starting service monitor..." "Cyan"
    $monitorLog = "$LogDir\monitor_$Timestamp.log"
    $monitorCmd = "cd '$ProjectRoot'; Write-Host '=== SERVICE MONITOR ===' -ForegroundColor Cyan; powershell -NoProfile -ExecutionPolicy Bypass -File '$ProjectRoot\monitor_services.ps1' 2>&1 | Tee-Object -FilePath '$monitorLog'"
    Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList "-NoExit", "-Command", $monitorCmd
    Write-ColorOutput "     Log: $monitorLog" "Gray"
}

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
