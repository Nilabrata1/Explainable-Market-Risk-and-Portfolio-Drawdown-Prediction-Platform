$ErrorActionPreference = "SilentlyContinue"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$PidFile = Join-Path $ProjectRoot "deploy_logs\pids.json"

if (-not (Test-Path $PidFile)) {
    Write-Host "No deploy_logs/pids.json file found. Nothing to stop."
    exit 0
}

$pids = Get-Content -Path $PidFile -Raw | ConvertFrom-Json

function Stop-ProcessTree {
    param([int]$RootProcessId)

    $children = Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $RootProcessId }
    foreach ($child in $children) {
        Stop-ProcessTree -RootProcessId $child.ProcessId
    }

    Stop-Process -Id $RootProcessId -Force
}

foreach ($pidValue in @($pids.api, $pids.streamlit, $pids.mlflow)) {
    if ($pidValue) {
        Stop-ProcessTree -RootProcessId $pidValue
    }
}

Remove-Item -Path $PidFile -Force
Write-Host "Stopped local deployment processes."
