# GPU Setup Notes

**Date:** November 25, 2025  
**Status:** CPU Mode (GPU optimization deferred)

---

## GPU Acceleration Attempt

### Goal
Enable CUDA GPU acceleration for llama-cpp-python to achieve 10-20x faster inference (from ~2 tokens/sec to ~20-40 tokens/sec).

### Hardware
- **GPU:** NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM)
- **System:** Windows 11, 16GB RAM
- **Confirmed:** torch.cuda.is_available() = True

### Issues Encountered

#### 1. Missing CUDA Toolkit
**Error:**
```
CMake Error: Could not find `nvcc` executable in any searched paths
CUDA Toolkit not found
```

**Root Cause:** NVIDIA CUDA Toolkit not installed on system.

**Required:** CUDA Toolkit 12.x (~3-4 GB download, ~10 GB installed)

**Download:** https://developer.nvidia.com/cuda-downloads

---

#### 2. Visual Studio 2026 Compatibility Issue
**Error:**
```
C:\Users\Sanjay Vishwa\AppData\Local\Temp\...\common.cpp(432,32): 
error C2039: 'system_clock': is not a member of 'std::chrono'
```

**Root Cause:** Visual Studio Build Tools 2026 (version 18, MSVC 19.50.35718.0) is too new for llama-cpp-python 0.3.2 compatibility.

**Details:**
- Compiler found at: `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe`
- Build Tools version: Visual Studio 18 (2026)
- llama-cpp-python version: 0.3.2
- Compilation failed with C++ standard library errors

**Potential Solutions:**
1. Install older Visual Studio 2022 Build Tools (may conflict)
2. Wait for llama-cpp-python update to support VS 2026
3. Use pre-built wheels (current solution)

---

### Current Solution: CPU Mode

**Status:** ✅ Working perfectly

**Performance:**
- Model load time: ~1.7 seconds
- Inference speed: ~2 tokens/sec
- Test 1: 296 tokens in 129 seconds (2.3 tok/sec)
- Test 2: 114 tokens in 58 seconds (1.9 tok/sec)
- Response quality: Excellent

**Installation:**
```powershell
# Installed pre-built CPU wheel
pip install "C:\Projects\Predictive Maintenance\LLM\scripts\setup\llama_cpp_python-0.3.2-cp311-cp311-win_amd64.whl"
```

**Verification:**
```powershell
python -c "from llama_cpp import Llama; print('llama-cpp-python working: CPU mode')"
# Output: llama-cpp-python working: CPU mode ✅
```

---

## Future GPU Setup (When Needed)

### Prerequisites
1. **Install CUDA Toolkit 12.x:**
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Select: Windows → x86_64 → Windows 10/11 → exe (local)
   - Size: ~3-4 GB download, ~10 GB installed
   - Time: 20-30 minutes

2. **Verify CUDA Installation:**
   ```powershell
   nvcc --version
   # Should show CUDA compilation tools version
   ```

3. **Rebuild llama-cpp-python with CUDA:**
   ```powershell
   # Set environment variables
   $env:CMAKE_ARGS="-DGGML_CUDA=on"
   $env:FORCE_CMAKE="1"
   
   # Uninstall CPU version
   pip uninstall llama-cpp-python -y
   
   # Rebuild with CUDA
   pip install llama-cpp-python --no-cache-dir --force-reinstall
   ```

4. **Test GPU Mode:**
   ```python
   from llama_cpp import Llama
   
   model = Llama(
       model_path="LLM/models/llama-3.1-8b-instruct-q4.gguf",
       n_gpu_layers=-1,  # Use all GPU layers
       n_ctx=2048,
       verbose=True  # Will show GPU layer offloading
   )
   ```

### Expected Benefits
- **Speed:** 20-40 tokens/sec (10-20x faster)
- **Production Readiness:** Real-time responses (<10 seconds)
- **Typical Response:** 296 tokens in 7-15 seconds (vs 129 seconds CPU)

---

## Recommendations

### For Development (Current Phase 3.1-3.4)
✅ **Use CPU mode** - Performance is acceptable for:
- Testing RAG retrieval
- Developing prompt templates
- Creating synthetic knowledge base
- Initial integration with ML models

### For Production (Phase 3.7+)
⚠️ **Consider GPU mode** if:
- Need real-time responses (<5 seconds)
- Processing multiple simultaneous requests
- Deploying as FastAPI server with concurrent users

### Alternative: Cloud GPU
If local GPU setup remains problematic:
- Use Raspberry Pi 5 (8GB) for edge deployment (CPU only)
- Offload LLM to cloud GPU for web interface
- Use RunPod, Vast.ai, or Lambda Labs (~$0.20-0.50/hour)

---

## Notes

**Current Status (Nov 25, 2025):**
- Phase 3.1: ✅ Complete (RAG infrastructure)
- Phase 3.2.1: ✅ Complete (Model installed)
- Phase 3.2.2: ✅ Complete (Inference working, CPU mode)
- Phase 3.2.3: 🔄 In Progress (CPU optimization, GPU deferred)

**Decision:** Continue with CPU mode for Phase 3.3-3.4 development. Revisit GPU setup before Phase 3.7 (Production Deployment) if performance requirements demand it.

**Performance Acceptable Because:**
1. Development/testing doesn't require real-time responses
2. Quality matters more than speed during prompt engineering
3. Can optimize later based on actual production needs
4. Raspberry Pi deployment will be CPU-only anyway
