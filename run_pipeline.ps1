param (
    [string]$DataDir = ".\data",
    [string]$TestFile = ".\data\on_state\on_merged.wav",
    [string]$ArtifactsDir = ".\artifacts"
)

# Configuration
$IMAGE_TRAINING = "sound-anomaly-training"
$IMAGE_INFERENCE = "sound-anomaly-inference"

# Resolve absolute paths
$DataDir = Resolve-Path $DataDir | Select-Object -ExpandProperty Path
$ArtifactsDir = $ArtifactsDir -replace "\\$", "" # Remove trailing slash
if (-not (Test-Path $ArtifactsDir)) {
    New-Item -ItemType Directory -Force -Path $ArtifactsDir | Out-Null
}
$ArtifactsDir = Resolve-Path $ArtifactsDir | Select-Object -ExpandProperty Path

# Validate Test File
if (-not (Test-Path $TestFile)) {
    Write-Error "Test file not found at $TestFile"
    exit 1
}
$TestFile = Resolve-Path $TestFile | Select-Object -ExpandProperty Path

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Data Dir:      $DataDir"
Write-Host "  Test File:     $TestFile"
Write-Host "  Artifacts Dir: $ArtifactsDir"
Write-Host "=================================================="

Write-Host "Step 1: Running Training..." -ForegroundColor Cyan
# Run training container
docker run --rm `
    -v "$($DataDir):/app/data" `
    -v "$($ArtifactsDir):/app/artifacts" `
    $IMAGE_TRAINING

if ($LASTEXITCODE -ne 0) {
    Write-Error "Training failed."
    exit 1
}

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Step 2: Locating New Model..." -ForegroundColor Cyan

# Find the most recently created directory in artifacts
$LatestDir = Get-ChildItem -Path $ArtifactsDir -Directory | Sort-Object CreationTime -Descending | Select-Object -First 1

if ($null -eq $LatestDir) {
    Write-Error "Error: No artifact directory found."
    exit 1
}

$ModelPathHost = Join-Path $LatestDir.FullName "sound_classifier.tflite"

if (-not (Test-Path $ModelPathHost)) {
    Write-Error "Error: Model file not found at $ModelPathHost"
    exit 1
}

Write-Host "Found latest model: $ModelPathHost" -ForegroundColor Green

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Step 3: Running Inference Verification..." -ForegroundColor Cyan

# Prepare test file mounting
$TestFileDir = Split-Path -Parent $TestFile
$TestFileName = Split-Path -Leaf $TestFile

# Run inference container
docker run --rm `
    -v "$($TestFileDir):/app/test_data" `
    -v "$($ModelPathHost):/app/model.tflite" `
    --entrypoint python `
    $IMAGE_INFERENCE `
    /app/test/test_docker_inference.py `
    "/app/test_data/$TestFileName" `
    "/app/model.tflite"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Pipeline Complete." -ForegroundColor Cyan
Write-Host "=================================================="
