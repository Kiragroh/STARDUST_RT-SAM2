import torch
import sys

def test_cuda():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your PyTorch installation.")
        sys.exit(1)
        
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Create a test tensor and move it to GPU
    x = torch.randn(2, 3)
    print("\nCPU Tensor:")
    print(x)
    
    # Move tensor to GPU
    x_cuda = x.cuda()
    print("\nGPU Tensor:")
    print(x_cuda)
    
    # Perform a simple operation
    y_cuda = x_cuda * 2
    print("\nGPU Computation Result (x * 2):")
    print(y_cuda)
    
    print("\nCUDA test completed successfully!")

if __name__ == "__main__":
    test_cuda()
