import torch

img2mse_torch = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr_torch = lambda x : 10. * torch.log10(1.0 / x)