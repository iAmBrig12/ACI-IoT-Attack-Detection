import torch
print(torch.version.cuda)  # Check which CUDA version PyTorch detects
print(torch.cuda.is_available())  # Check if CUDA is available
print(torch.backends.cudnn.enabled)  # Check if cuDNN is enabled (optional)
