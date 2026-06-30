param(
    [string]$HostAddress = "127.0.0.1",
    [int]$ApiPort = 8000,
    [int]$StreamlitPort = 8501,
    [int]$MlflowPort = 5000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Logs = Join-Path $ProjectRoot "deploy_logs"
$PidFile = Join-Path $Logs "pids.json"

if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Create it and install requirements before deploying."
}

New-Item -ItemType Directory -Force -Path $Logs | Out-Null

$api = Start-Process -FilePath $Python `
    -ArgumentList "-m","uvicorn","api.main:app","--host",$HostAddress,"--port",$ApiPort `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $Logs "api.out.log") `
    -RedirectStandardError (Join-Path $Logs "api.err.log") `
    -PassThru

$streamlit = Start-Process -FilePath $Python `
    -ArgumentList "-m","streamlit","run","streamlit_app/app.py","--server.address",$HostAddress,"--server.port",$StreamlitPort,"--server.headless","true" `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $Logs "streamlit.out.log") `
    -RedirectStandardError (Join-Path $Logs "streamlit.err.log") `
    -PassThru

$mlflowCommand = "`$env:MLFLOW_ALLOW_FILE_STORE='true'; & '$Python' -m mlflow ui --backend-store-uri ./mlruns --host $HostAddress --port $MlflowPort"
$mlflow = Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-Command",$mlflowCommand `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $Logs "mlflow.out.log") `
    -RedirectStandardError (Join-Path $Logs "mlflow.err.log") `
    -PassThru

@{
    api = $api.Id
    streamlit = $streamlit.Id
    mlflow = $mlflow.Id
    started_at = (Get-Date).ToString("s")
} | ConvertTo-Json | Set-Content -Path $PidFile -Encoding UTF8

Start-Sleep -Seconds 6

Write-Host "FastAPI:   http://$HostAddress`:$ApiPort"
Write-Host "Streamlit: http://$HostAddress`:$StreamlitPort"
Write-Host "MLflow:    http://$HostAddress`:$MlflowPort"
Write-Host "Logs:      $Logs"

