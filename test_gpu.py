import torch
import os

# Set OpenMP environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("=== GPU/CUDA Test ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name()}")
    print(f"CUDA device capability: {torch.cuda.get_device_capability()}")
    
    # Test GPU tensor creation
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print("✓ GPU tensor operations working")
        print(f"Result shape: {z.shape}")
    except Exception as e:
        print(f"✗ GPU tensor operations failed: {e}")
else:
    print("CUDA not available")

print("\n=== Device Selection Test ===")
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)
print(f"Selected device: {device}")

# Test with a simple tensor
test_tensor = torch.randn(10, 10)
print(f"Test tensor device: {test_tensor.device}")

if device == "cuda":
    test_tensor = test_tensor.cuda()
    print(f"Test tensor moved to GPU: {test_tensor.device}") 