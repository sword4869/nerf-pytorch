
import numpy as np
from PIL import Image
import torch


def save_image(img, path):
    img = img.detach().cpu().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


class ImgBuffer:
    def __init__(self, H, W):
        self.H = H
        self.W = W
        self.count = 0
        self.buffer = None

    def update(self, input, path):
        self.buffer = torch.concat([self.buffer, input], 0) if self.buffer is not None else input
        if len(self.buffer) >= self.H * self.W:
            img, self.buffer = torch.split(self.buffer, [self.H*self.W, len(self.buffer) - self.H*self.W], dim=0)
            img = img.reshape(self.H, self.W, 3)
            save_image(img, f'{path}_{self.count}.png')
            self.count += 1