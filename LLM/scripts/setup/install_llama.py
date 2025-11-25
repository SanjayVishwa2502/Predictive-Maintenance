"""
Download Llama 3.1 8B GGUF (llama.cpp format)
Phase 3.2.1: Model Installation
No transformers needed!
"""
import urllib.request
from pathlib import Path
import sys
import time

def download_llama_31_8b_gguf():
    """Download pre-quantized GGUF model"""
    
    # Use absolute path to avoid directory issues
    project_root = Path(__file__).resolve().parents[3]
    output_dir = project_root / "LLM" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Q4_K_M = 4-bit quantized, medium quality
    model_url = "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    model_file = output_dir / "llama-3.1-8b-instruct-q4.gguf"
    
    # Check if already downloaded
    if model_file.exists():
        file_size_gb = model_file.stat().st_size / 1e9
        print(f"✓ Model already exists: {model_file}")
        print(f"✓ Size: {file_size_gb:.2f} GB")
        
        if file_size_gb < 4.5:  # Expected size is ~4.9 GB
            print("\n⚠ Warning: File size seems incomplete. Re-downloading...")
            model_file.unlink()
        else:
            print("\n✓ Model ready to use!")
            print(f"✓ VRAM usage: ~3 GB (5GB headroom on 8GB card!)")
            print(f"\nNext step: Install llama-cpp-python")
            print(f"  pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
            return
    
    print("="*60)
    print("Phase 3.2.1: Downloading Llama 3.1 8B (GGUF Q4_K_M)")
    print("="*60)
    print(f"Source: HuggingFace (bartowski)")
    print(f"Size: ~4.9 GB")
    print(f"URL: {model_url}")
    print(f"Destination: {model_file}")
    print("\nThis may take 10-20 minutes depending on your connection...")
    print("Please do NOT close this window during download.\n")
    
    def show_progress(block_num, block_size, total_size):
        """Display download progress"""
        downloaded = block_num * block_size
        percent = min(downloaded / total_size * 100, 100)
        downloaded_gb = downloaded / 1e9
        total_gb = total_size / 1e9
        
        # Progress bar
        bar_length = 40
        filled = int(bar_length * downloaded / total_size)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r[{bar}] {percent:.1f}% ({downloaded_gb:.2f}/{total_gb:.2f} GB)", end="", flush=True)
    
    try:
        print("Starting download with retry logic...")
        
        # Try download with multiple attempts
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    print(f"\nRetry attempt {attempt}/{max_retries}...")
                    time.sleep(2)
                
                urllib.request.urlretrieve(model_url, model_file, show_progress)
                print("\n")  # New line after progress
                break  # Success!
                
            except Exception as download_error:
                if attempt == max_retries:
                    raise download_error
                print(f"\n⚠ Download interrupted: {download_error}")
                if model_file.exists():
                    model_file.unlink()
                print("Retrying...")
        
        # Verify download
        file_size = model_file.stat().st_size
        file_size_gb = file_size / 1e9
        
        print("="*60)
        print("Download Complete!")
        print("="*60)
        print(f"✓ Model saved to: {model_file}")
        print(f"✓ Size: {file_size_gb:.2f} GB")
        print(f"✓ File exists: {model_file.exists()}")
        
        # Validate size
        if file_size_gb < 4.5:
            print(f"\n⚠ Warning: File size ({file_size_gb:.2f} GB) is smaller than expected (~4.9 GB)")
            print("Download may be incomplete. Please run again.")
            sys.exit(1)
        
        print(f"\n✓ VRAM usage: ~3 GB (5GB headroom on RTX 4070 8GB!)")
        print(f"✓ Inference speed: 1-2 sec for 100-200 tokens (FAST!)")
        print(f"✓ No transformers dependency!")
        
        print("\n" + "="*60)
        print("Next Step: Install llama-cpp-python")
        print("="*60)
        print("Run the following command:")
        print("  pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
        print("\nOr if you don't have CUDA 12.1, use CPU version:")
        print("  pip install llama-cpp-python")
        print("\n" + "="*60)
        print("Phase 3.2.1 Complete: Model Downloaded Successfully!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Download interrupted by user.")
        if model_file.exists():
            model_file.unlink()
            print("✓ Partial file removed.")
        print("Run script again to resume download.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\n✗ Error during download: {e}")
        if model_file.exists():
            model_file.unlink()
            print("✓ Partial file removed.")
        print("Please check your internet connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    download_llama_31_8b_gguf()
