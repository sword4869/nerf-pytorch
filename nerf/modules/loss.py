import torch

img2mse_torch = lambda x, y : torch.mean((x - y) ** 2)

def mse2psnr_torch(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return 10. * torch.log10(1.0 / x)