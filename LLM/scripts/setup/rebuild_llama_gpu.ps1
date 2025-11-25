# Rebuild llama-cpp-python with CUDA GPU support
# Run this after installing Visual Studio Build Tools

Write-Host "="*60
Write-Host "Rebuilding llama-cpp-python with GPU Support"
Write-Host "="*60

# Activate venv
& "C:/Projects/Predictive Maintenance/venv/Scripts/Activate.ps1"

Write-Host "`n1. Checking for Visual Studio Build Tools..."
$vsPath = Get-ChildItem "C:\Program Files\Microsoft Visual Studio\" -Recurse -Filter "cl.exe" -ErrorAction SilentlyContinue | Select-Object -First 1

if (-not $vsPath) {
    Write-Host "✗ Visual Studio Build Tools not found."
    Write-Host "Please restart PowerShell and run this script again."
    pause
    exit 1
}

Write-Host "✓ Found: $($vsPath.Directory.FullName)"

# Import Visual Studio environment
Write-Host "`n2. Setting up Visual Studio environment..."
$vcvarsPath = Get-ChildItem "C:\Program Files\Microsoft Visual Studio\" -Recurse -Filter "vcvars64.bat" -ErrorAction SilentlyContinue | Select-Object -First 1

if ($vcvarsPath) {
    Write-Host "✓ Found vcvars64.bat"
    cmd /c "`"$($vcvarsPath.FullName)`" && set" | ForEach-Object {
        if ($_ -match "^(.*?)=(.*)$") {
            Set-Content "env:\$($matches[1])" $matches[2]
        }
    }
} else {
    Write-Host "⚠ vcvars64.bat not found, trying direct path..."
}

Write-Host "`n3. Uninstalling current llama-cpp-python..."
pip uninstall llama-cpp-python -y

Write-Host "`n4. Setting CUDA environment variables..."
$env:CMAKE_ARGS = "-DLLAMA_CUBLAS=on"
$env:FORCE_CMAKE = "1"

Write-Host "`n5. Installing llama-cpp-python with CUDA support..."
Write-Host "This may take 5-10 minutes to compile..."
pip install llama-cpp-python --no-cache-dir --force-reinstall --verbose

Write-Host "`n6. Verifying installation..."
python -c "import llama_cpp; print('✓ llama-cpp-python installed'); print(f'Version: {llama_cpp.__version__}')"

Write-Host "`n="*60
Write-Host "Installation Complete!"
Write-Host "="*60
Write-Host "`nNext: Run test_llama.py to verify GPU acceleration"
Write-Host "Expected: 20-40 tokens/sec (vs 2 tokens/sec CPU mode)"
pause
