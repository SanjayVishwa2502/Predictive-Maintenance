# Predictive Maintenance Dashboard - Optimized Shutdown Script
# This PowerShell script stops all running services with port-based detection
# Optimized for: Accurate process targeting, graceful shutdown, cleanup verification
# Phase 3.7.3 - Production-Ready Stack

# ============================================================================
# CONFIGURATION
# ============================================================================
$ErrorActionPreference = "Continue"

function Write-ColorOutput($Message, $Color = "White") {
    Write-Host $Message -ForegroundColor $Color
}

# ============================================================================
# SHUTDOWN BANNER
# ============================================================================
Clear-Host
Write-ColorOutput "========================================" "Red"
Write-ColorOutput " Stopping Dashboard Services" "Red"
Write-ColorOutput "========================================" "Red"
Write-ColorOutput ""

# ============================================================================
# PORT-BASED PROCESS TERMINATION (More Accurate)
# ============================================================================
Write-ColorOutput "[SHUTDOWN] Stopping services by port..." "Yellow"

$ports = @(
    @{Port=8000; Name="Backend (FastAPI)"},
    @{Port=5555; Name="Flower Monitor"},
    @{Port=5173; Name="Frontend (Vite)"}
)

$stoppedCount = 0
foreach ($portInfo in $ports) {
    $port = $portInfo.Port
    $name = $portInfo.Name
    
    try {
        $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
        if ($connections) {
            $processIds = $connections | Select-Object -ExpandProperty OwningProcess -Unique
            foreach ($procId in $processIds) {
                $process = Get-Process -Id $procId -ErrorAction SilentlyContinue
                if ($process) {
                    Write-ColorOutput "  [PORT $port] Stopping $name (PID: $procId)..." "White"
                    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                    Start-Sleep -Milliseconds 500
                    Write-ColorOutput "     [OK] $name stopped" "Green"
                    $stoppedCount++
                }
            }
        } else {
            Write-ColorOutput "  [PORT $port] $name not running" "Gray"
        }
    } catch {
        Write-ColorOutput "  [PORT $port] [WARN] Error checking $name" "Yellow"
    }
}

Write-ColorOutput ""

# Stop printer datalogger processes (no port binding)
try {
    $loggerProcs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*3Dprinterdata*datalogger.py*" }
    if ($loggerProcs) {
        foreach ($p in ($loggerProcs | Select-Object -Unique ProcessId)) {
            Write-ColorOutput "  [LOGGER] Stopping datalogger (PID: $($p.ProcessId))..." "White"
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
            Start-Sleep -Milliseconds 300
            Write-ColorOutput "     [OK] datalogger stopped" "Green"
        }
    } else {
        Write-ColorOutput "  [LOGGER] datalogger not running" "Gray"
    }
} catch {
    Write-ColorOutput "  [LOGGER] [WARN] Error stopping datalogger" "Yellow"
}

# ============================================================================
# FALLBACK: PROCESS NAME CLEANUP
# ============================================================================
Write-ColorOutput "[CLEANUP] Additional process cleanup..." "Yellow"

# Kill remaining node processes (Frontend)
$nodeProcesses = Get-Process -Name "node" -ErrorAction SilentlyContinue
if ($nodeProcesses) {
    $nodeCount = $nodeProcesses.Count
    Stop-Process -Name "node" -Force -ErrorAction SilentlyContinue
    Write-ColorOutput "  [-] Killed $nodeCount Node.js process(es)" "Gray"
    $stoppedCount += $nodeCount
}

# Kill remaining Python processes (Backend, Celery, Flower)
$pythonProcesses = Get-Process -Name "python" -ErrorAction SilentlyContinue
if ($pythonProcesses) {
    # Filter to only kill processes in project directory
    $projectProcesses = $pythonProcesses | Where-Object {
        $_.Path -like "*Predictive Maintenance*"
    }
    if ($projectProcesses) {
        $pythonCount = $projectProcesses.Count
        $projectProcesses | Stop-Process -Force -ErrorAction SilentlyContinue
        Write-ColorOutput "  [-] Killed $pythonCount Python process(es)" "Gray"
        $stoppedCount += $pythonCount
    }
}

# Additional specific process cleanup
$cleanupProcesses = @("uvicorn", "celery", "flower")
foreach ($procName in $cleanupProcesses) {
    $proc = Get-Process -Name $procName -ErrorAction SilentlyContinue
    if ($proc) {
        Stop-Process -Name $procName -Force -ErrorAction SilentlyContinue
        Write-ColorOutput "  [-] Killed $procName" "Gray"
    }
}

# Kill orphaned PowerShell windows (hidden background processes)
$currentPid = $PID
$allPowershell = Get-Process -Name "powershell" -ErrorAction SilentlyContinue | 
                 Where-Object { $_.Id -ne $currentPid }
foreach ($ps in $allPowershell) {
    # Check if it's running dashboard-related commands
    $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($ps.Id)" -ErrorAction SilentlyContinue).CommandLine
    if ($cmdLine -like "*uvicorn*" -or $cmdLine -like "*celery*" -or $cmdLine -like "*npm run dev*") {
        Stop-Process -Id $ps.Id -Force -ErrorAction SilentlyContinue
        Write-ColorOutput "  [-] Killed background PowerShell (PID: $($ps.Id))" "Gray"
        $stoppedCount++
    }
}

Write-ColorOutput ""

# ============================================================================
# VERIFICATION
# ============================================================================
Write-ColorOutput "[VERIFY] Checking if ports are free..." "Cyan"

$allClear = $true
foreach ($portInfo in $ports) {
    $port = $portInfo.Port
    $connection = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($connection) {
        Write-ColorOutput "  [WARN] Port $port still in use!" "Yellow"
        $allClear = $false
    } else {
        Write-ColorOutput "  [OK] Port $port is free" "Green"
    }
}

Write-ColorOutput ""

# ============================================================================
# SUMMARY
# ============================================================================
if ($allClear) {
    Write-ColorOutput "========================================" "Green"
    Write-ColorOutput " [OK] All services stopped successfully!" "Green"
    Write-ColorOutput "========================================" "Green"
} else {
    Write-ColorOutput "========================================" "Yellow"
    Write-ColorOutput " [WARN] Some ports still in use" "Yellow"
    Write-ColorOutput "========================================" "Yellow"
    Write-ColorOutput ""
    Write-ColorOutput "  Try running this script again or manually:" "Gray"
    Write-ColorOutput "  Get-NetTCPConnection -LocalPort 8000,5555,5173 | Select OwningProcess | Stop-Process -Force" "Gray"
}

Write-ColorOutput ""
Write-ColorOutput "Statistics:" "White"
Write-ColorOutput "  Processes stopped: $stoppedCount" "Cyan"

Write-ColorOutput ""
Write-ColorOutput "System Services Status:" "Yellow"
Write-ColorOutput "  PostgreSQL and Redis services remain running (as designed)" "Gray"
Write-ColorOutput "  To stop them: Stop-Service postgresql-x64-18; Stop-Service Redis" "Gray"

Write-ColorOutput ""
Write-ColorOutput "[OK] Shutdown complete. Safe to close this window." "Green"
Write-ColorOutput ""
