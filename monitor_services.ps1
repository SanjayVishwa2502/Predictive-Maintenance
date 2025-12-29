# Monitor Predictive Maintenance services (API/Frontend/Flower)
# Logs state transitions (up/down) to logs/monitor_*.log

$ErrorActionPreference = "Continue"

$ProjectRoot = "C:\Projects\Predictive Maintenance"
$LogDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

$Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$LogFile = Join-Path $LogDir "monitor_$Timestamp.log"

$ApiUrl = "http://localhost:8000/api/ml/health"
$PollSeconds = 2

function Write-Log($msg) {
  $line = "$(Get-Date -Format o)  $msg"
  $line | Tee-Object -FilePath $LogFile -Append | Out-Host
}

function Test-PortUp($port) {
  try {
    return (Test-NetConnection -ComputerName localhost -Port $port -WarningAction SilentlyContinue).TcpTestSucceeded
  } catch {
    return $false
  }
}

function Test-ApiUp() {
  try {
    $r = Invoke-RestMethod -Uri $ApiUrl -TimeoutSec 3
    if ($null -ne $r -and $r.status) { return @{ up=$true; status=$r.status } }
    return @{ up=$true; status="unknown" }
  } catch {
    return @{ up=$false; status=$null }
  }
}

Write-Log "[MONITOR] Starting service monitor"
Write-Log "[MONITOR] API: $ApiUrl"
Write-Log "[MONITOR] Log: $LogFile"

$prev = @{ api=$null; apiStatus=$null; p8000=$null; p5173=$null; p5555=$null }

while ($true) {
  $p8000 = Test-PortUp 8000
  $p5173 = Test-PortUp 5173
  $p5555 = Test-PortUp 5555
  $api = Test-ApiUp

  if ($prev.p8000 -ne $p8000) { Write-Log "[PORT 8000] " + ($(if($p8000){'UP'} else {'DOWN'})) }
  if ($prev.p5173 -ne $p5173) { Write-Log "[PORT 5173] " + ($(if($p5173){'UP'} else {'DOWN'})) }
  if ($prev.p5555 -ne $p5555) { Write-Log "[PORT 5555] " + ($(if($p5555){'UP'} else {'DOWN'})) }

  if ($prev.api -ne $api.up) {
    Write-Log "[API] " + ($(if($api.up){'UP'} else {'DOWN'}))
  }
  if ($api.up -and $prev.apiStatus -ne $api.status) {
    Write-Log "[API] status=" + $api.status
  }

  $prev.p8000 = $p8000
  $prev.p5173 = $p5173
  $prev.p5555 = $p5555
  $prev.api = $api.up
  $prev.apiStatus = $api.status

  Start-Sleep -Seconds $PollSeconds
}
