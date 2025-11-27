$ErrorActionPreference = "Stop"

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "       SETTING UP GPU ACCELERATION (CUDA 12.4)     " -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

# 1. Uninstall existing package
Write-Host "`n[1/3] Uninstalling existing llama-cpp-python..." -ForegroundColor Yellow
pip uninstall -y llama-cpp-python
if ($LASTEXITCODE -ne 0) { Write-Host "Warning: Uninstall failed or package not found. Continuing..." -ForegroundColor Yellow }

# 2. Install CUDA-enabled version
Write-Host "`n[2/3] Installing CUDA-enabled llama-cpp-python and dependencies..." -ForegroundColor Yellow
Write-Host "Targeting: CUDA 12.1 (Pinned version 0.3.4 for Windows compatibility)" -ForegroundColor Gray

# Install NVIDIA runtime libraries to provide missing DLLs
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12

# Install llama-cpp-python with CUDA support (pinning to 0.3.4 which has Windows wheels)
pip install llama-cpp-python==0.3.4 --force-reinstall --no-cache-dir --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[3/3] Installation SUCCESS!" -ForegroundColor Green
} else {
    Write-Host "`n[3/3] Installation FAILED!" -ForegroundColor Red
    exit 1
}

Write-Host "`nNow running verification script..." -ForegroundColor Cyan

# Add NVIDIA DLL paths to PATH for this session
try {
    $pythonPath = python -c "import sys; print(sys.exec_prefix)"
    $nvidiaBase = Join-Path $pythonPath "Lib\site-packages\nvidia"
    $cublasBin = Join-Path $nvidiaBase "cublas\bin"
    $cudaRuntimeBin = Join-Path $nvidiaBase "cuda_runtime\bin"

    if (Test-Path $cublasBin) {
        $env:PATH = "$cublasBin;$env:PATH"
        Write-Host "Added to PATH: $cublasBin" -ForegroundColor Gray
    }
    if (Test-Path $cudaRuntimeBin) {
        $env:PATH = "$cudaRuntimeBin;$env:PATH"
        Write-Host "Added to PATH: $cudaRuntimeBin" -ForegroundColor Gray
    }
} catch {
    Write-Host "Warning: Could not automatically add NVIDIA DLLs to PATH. You may need to add them manually." -ForegroundColor Yellow
}

python "C:\Projects\Predictive Maintenance\LLM\scripts\inference\verify_gpu.py"