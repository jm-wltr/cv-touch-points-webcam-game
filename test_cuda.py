import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    x = torch.rand(3, 3).cuda()
    print("Tensor on:", x.device)
