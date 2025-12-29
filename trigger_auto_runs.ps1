param(
  [string]$BaseUrl = "http://localhost:8000",
  [int]$IntervalSeconds = 150,
  [string]$MachineId = "",
  [switch]$RunOnce
)

$ErrorActionPreference = "Stop"

function Invoke-JsonGet {
  param([string]$Url)
  return Invoke-RestMethod -Uri $Url -Method GET -Headers @{ "Accept" = "application/json" }
}

function Invoke-JsonPost {
  param([string]$Url)
  return Invoke-RestMethod -Uri $Url -Method POST -Headers @{ "Accept" = "application/json"; "Content-Type" = "application/json" }
}

if ($IntervalSeconds -lt 30 -or $IntervalSeconds -gt 3600) {
  throw "IntervalSeconds must be between 30 and 3600"
}

$machinesUrl = "$BaseUrl/api/ml/machines"

$machineIds = @()
if ($MachineId.Trim()) {
  $machineIds = @($MachineId.Trim())
} else {
  $machinesResp = Invoke-JsonGet -Url $machinesUrl
  if (-not $machinesResp -or -not $machinesResp.machines) {
    throw "No machines returned from $machinesUrl"
  }
  $machineIds = @($machinesResp.machines | ForEach-Object { $_.machine_id } | Where-Object { $_ -and $_.ToString().Trim() })
}

Write-Host ("Machines {0}" -f $machineIds.Count)

foreach ($mid in $machineIds) {
  try {
    $midEscaped = [Uri]::EscapeDataString($mid)
    if ($RunOnce) {
      $url = "$BaseUrl/api/ml/machines/$midEscaped/auto/run_once"
      $resp = Invoke-JsonPost -Url $url
      Write-Host "OK run_once $mid -> run_id=$($resp.run_id) data_stamp=$($resp.data_stamp)"
      continue
    }

    $url = "$BaseUrl/api/ml/machines/$midEscaped/auto/start?interval_seconds=$IntervalSeconds"
    $resp = Invoke-JsonPost -Url $url
    Write-Host "OK auto/start $mid -> running=$($resp.running) interval=$($resp.interval_seconds)"
  } catch {
    Write-Host ("FAIL {0} - {1}" -f $mid, $_.Exception.Message)
  }
}

Write-Host "Done."
