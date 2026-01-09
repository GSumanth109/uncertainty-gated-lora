import torch
import sys

def check():
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✅ Success: Running on NVIDIA GPU ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        print(f"✅ Success: Running on Apple Silicon (Metal Performance Shaders)")
    else:
        print(f"⚠️ Warning: No GPU detected. Running on CPU.")

if __name__ == "__main__":
    check()
