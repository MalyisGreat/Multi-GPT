param(
    [string]$PythonExe = "C:/Users/joshj/AppData/Local/Programs/Python/Python312/python.exe",
    [string]$VenvName = ".venv312",
    [string]$DatasetDir = "data/simple_cifar10_caption",
    [string]$OutputDir = "checkpoints/full-run",
    [int]$TrainSize = 8000,
    [int]$ValSize = 1000,
    [int]$HeldoutSize = 1000,
    [int]$Epochs = 10,
    [int]$BatchSize = 32,
    [int]$GradAccum = 2,
    [int]$NumWorkers = 8,
    [string]$Precision = "fp16"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "[1/4] Creating environment + installing deps..."
powershell -ExecutionPolicy Bypass -File "$root/scripts/bootstrap.ps1" -PythonExe $PythonExe -VenvName $VenvName -EnableCuda

$pythonPath = Join-Path $root "$VenvName/Scripts/python.exe"
if (-not (Test-Path $pythonPath)) {
    throw "Python interpreter not found at $pythonPath"
}

Write-Host "[2/4] Building dataset (auto-download if needed)..."
& $pythonPath src/data/prepare_simple_cifar10_dataset.py `
    --output-dir $DatasetDir `
    --train-size $TrainSize `
    --val-size $ValSize `
    --heldout-size $HeldoutSize `
    --overwrite

Write-Host "[3/4] Training model..."
& $pythonPath src/train_native_caption.py `
    --train-jsonl "$DatasetDir/train.jsonl" `
    --val-jsonl "$DatasetDir/val.jsonl" `
    --heldout-jsonl "$DatasetDir/heldout.jsonl" `
    --output-dir $OutputDir `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --gradient-accumulation-steps $GradAccum `
    --num-workers $NumWorkers `
    --precision $Precision

Write-Host "[4/4] Running multimodal benchmark..."
& $pythonPath src/benchmark_multimodal.py `
    --checkpoint-dir $OutputDir `
    --manifest "$DatasetDir/heldout.jsonl"

Write-Host "Full run complete."
Write-Host "Training artifacts: $OutputDir"
Write-Host "Benchmark report: $OutputDir/benchmark_report.json"
