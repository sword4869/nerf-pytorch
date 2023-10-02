import torch

class PositionalEmbedder():
    def __init__(self, multires, include_input=True, log_sampling=True):
        if log_sampling:
            self.freq_bands = 2 ** torch.linspace(0, multires - 1, steps=multires)  # [multires]
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** (multires - 1), steps=multires)

        self.include_input = include_input
        self.embed_num = 2 * multires + 1 if include_input else 2 * multires

    def __call__(self, x):
        for i, freq in enumerate(self.freq_bands.to(x.device)):
            if i == 0:
                y = torch.concat([torch.sin(x * freq), torch.cos(x * freq)], dim=-1)
            else:
                y = torch.concat([y, torch.sin(x * freq), torch.cos(x * freq)], dim=-1)

        if self.include_input:
            y = torch.cat([x, y], dim=-1)

        return y

    def cal_outdim(self, input_dims):
        return input_dims * self.embed_num
    
class FourierEmbedder():
    def __init__(self, multires, include_input=True, log_sampling=True):
        if log_sampling:
            self.freq_bands = 2 ** torch.linspace(0, multires - 1, steps=multires)  # [multires]
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** (multires - 1), steps=multires)

        self.include_input = include_input
        self.embed_num = 2 * multires + 1 if include_input else 2 * multires

    def __call__(self, x):
        for i, freq in enumerate(self.freq_bands.to(x.device)):
            if i == 0:
                y = torch.concat([torch.sin(x * freq), torch.cos(x * freq)], dim=-1)
            else:
                y = torch.concat([y, torch.sin(x * freq), torch.cos(x * freq)], dim=-1)

        if self.include_input:
            y = torch.cat([x, y], dim=-1)

        return y

    def cal_outdim(self, input_dims):
        return input_dims * self.embed_num