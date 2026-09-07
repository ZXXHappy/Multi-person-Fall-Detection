import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    x = torch.ones(1).cuda()
    print(f"Tensor is on: {x.device}")