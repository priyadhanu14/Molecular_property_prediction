param(
    [string]$VenvPath = ".venv",
    [string]$PythonCmd = "py",
    [string]$PythonVersion = "3.11",
    [string]$DataPath = "data/tox21_nr_ar.csv",
    [string]$Model = "attentivefp",
    [int]$Epochs = 60,
    [int]$BatchSize = 256,
    [string]$ExperimentName = "molecular-property-prediction",
    [switch]$SkipInstall,
    [switch]$SkipTrain,
    [switch]$NoDownload,
    [switch]$Headless
)

$ErrorActionPreference = "Stop"
$env:PYTHONINSPECT = ""

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Ensure-Command {
    param([string]$CommandName)
    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "Command '$CommandName' is not available in PATH."
    }
}

function Invoke-Checked {
    param(
        [string]$Exe,
        [string[]]$Args,
        [string]$StepName = "Command"
    )
    & $Exe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE."
    }
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $ProjectRoot) {
    $ProjectRoot = (Get-Location).Path
}
Set-Location $ProjectRoot

Write-Step "Using project root: $ProjectRoot"
Ensure-Command -CommandName $PythonCmd

$PythonPrefixArgs = @()
if ($PythonCmd -eq "py" -and $PythonVersion) {
    $PythonPrefixArgs += "-$PythonVersion"
}

try {
    $DetectedPythonVersion = (& $PythonCmd @PythonPrefixArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
}
catch {
    $DetectedPythonVersion = ""
}

if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($DetectedPythonVersion)) {
    Write-Host ""
    Write-Host "No suitable Python runtime found via '$PythonCmd $($PythonPrefixArgs -join ' ')'." -ForegroundColor Red
    Write-Host "Try one of:" -ForegroundColor Yellow
    Write-Host "  1) Install Python 3.11, then re-run:" -ForegroundColor Yellow
    Write-Host "     .\\run_demo.ps1 -PythonCmd py -PythonVersion 3.11" -ForegroundColor Yellow
    Write-Host "  2) If you have python.exe in PATH, run:" -ForegroundColor Yellow
    Write-Host "     .\\run_demo.ps1 -PythonCmd python" -ForegroundColor Yellow
    Write-Host "  3) List available runtimes:" -ForegroundColor Yellow
    Write-Host "     py --list" -ForegroundColor Yellow
    throw "Python runtime not available."
}

if ([version]$DetectedPythonVersion -lt [version]"3.9" -or [version]$DetectedPythonVersion -ge [version]"3.13") {
    throw "Detected Python $DetectedPythonVersion. Use Python 3.9-3.12 (recommended 3.11) because RDKit/PyG wheels are often unavailable for 3.13+."
}

$VenvPython = Join-Path $ProjectRoot "$VenvPath\Scripts\python.exe"
$VenvPip = Join-Path $ProjectRoot "$VenvPath\Scripts\pip.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Step "Creating virtual environment at $VenvPath"
    Invoke-Checked -Exe $PythonCmd -Args ($PythonPrefixArgs + @("-m", "venv", $VenvPath)) -StepName "Virtual environment creation"
}
else {
    Write-Step "Virtual environment already exists at $VenvPath"
}

if (-not $SkipInstall) {
    Write-Step "Installing dependencies"
    try {
        Invoke-Checked -Exe $VenvPython -Args @("-m", "pip", "install", "--upgrade", "pip") -StepName "pip upgrade"
        Invoke-Checked -Exe $VenvPip -Args @("install", "-r", "requirements.txt") -StepName "Dependency installation"
    }
    catch {
        Write-Host ""
        Write-Host "Dependency installation failed." -ForegroundColor Red
        Write-Host "If this is a Python version issue, recreate the venv with Python 3.11 and rerun." -ForegroundColor Yellow
        Write-Host "Example: .\run_demo.ps1 -PythonCmd py -PythonVersion 3.11" -ForegroundColor Yellow
        Write-Host "If this is a PyTorch/PyG wheel mismatch, install torch + matching PyG wheels first, then rerun with -SkipInstall." -ForegroundColor Yellow
        throw
    }
}
else {
    Write-Step "Skipping dependency installation (-SkipInstall)"
}

if (-not $SkipTrain) {
    Write-Step "Training model ($Model)"
    $TrainArgs = @(
        "train.py",
        "--data-path", $DataPath,
        "--model", $Model,
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--experiment-name", $ExperimentName
    )
    if (-not $NoDownload) {
        $TrainArgs += "--download-tox21"
    }
    Invoke-Checked -Exe $VenvPython -Args $TrainArgs -StepName "Model training"
}
else {
    Write-Step "Skipping training (-SkipTrain)"
}

$LatestCheckpoint = Get-ChildItem -Path (Join-Path $ProjectRoot "models") -Filter "*_best.pt" -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

Write-Step "Starting MLflow UI and Streamlit app"
$MlflowArgs = @(
    "-m", "mlflow", "ui",
    "--backend-store-uri", "./mlruns",
    "--host", "127.0.0.1",
    "--port", "5000"
)
$StreamlitArgs = @(
    "-m", "streamlit", "run", "app.py",
    "--server.address", "127.0.0.1",
    "--server.port", "8501"
)
if ($Headless) {
    $StreamlitArgs += @("--server.headless", "true")
}

Start-Process -FilePath $VenvPython -ArgumentList $MlflowArgs -WorkingDirectory $ProjectRoot | Out-Null
Start-Process -FilePath $VenvPython -ArgumentList $StreamlitArgs -WorkingDirectory $ProjectRoot | Out-Null

Write-Host ""
Write-Host "MLflow UI:   http://127.0.0.1:5000" -ForegroundColor Green
Write-Host "Streamlit:   http://127.0.0.1:8501" -ForegroundColor Green
if ($LatestCheckpoint) {
    Write-Host "Checkpoint:  models/$($LatestCheckpoint.Name)" -ForegroundColor Green
    Write-Host "Use that checkpoint path in the Streamlit sidebar."
}
else {
    Write-Host "No checkpoint found in ./models yet. Train once, then set checkpoint path in the Streamlit sidebar." -ForegroundColor Yellow
}
