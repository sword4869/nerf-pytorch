
import numpy as np
from PIL import Image
import torch


def save_image(img, path):
    img = img.detach().cpu().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


class ImgBuffer:
    def __init__(self, H, W, channel=3):
        self.H = H
        self.W = W
        self.channel = channel
        self.count = 0
        self.buffer = None

    def update(self, input, path):
        self.buffer = torch.concat([self.buffer, input], 0) if self.buffer is not None else input
        if len(self.buffer) >= self.H * self.W:
            img, self.buffer = torch.split(self.buffer, [self.H*self.W, len(self.buffer) - self.H*self.W], dim=0)
            if self.channel == 1:
                img = img.reshape(self.H, self.W)   
            else:
                img = img.reshape(self.H, self.W, self.channel)
            save_image(img, f'{path}_{self.count}.png')
            self.count += 1


class ImgBufferHub:
    def __init__(self, H, W) -> None:
        self.val_buffers = {
            'rgb_original': ImgBuffer(H, W),
            'rgb_coarse': ImgBuffer(H, W),
            'rgb_fine': ImgBuffer(H, W),
            'disp_coarse': ImgBuffer(H, W, 1),
            'disp_fine': ImgBuffer(H, W, 1),
        }
        self.test_buffers = {
            'rgb_original': ImgBuffer(H, W),
            'rgb_coarse': ImgBuffer(H, W),
            'rgb_fine': ImgBuffer(H, W),
            'disp_coarse': ImgBuffer(H, W, 1),
            'disp_fine': ImgBuffer(H, W, 1),
        }

    def update(self, path, **buffers):
        for key, buffer in buffers.items():
            if path.split('/')[-1].startswith('val'):
                self.val_buffers[key].update(buffer, f'{path}/{key}')
            elif path.split('/')[-1].startswith('test'):
                self.test_buffers[key].update(buffer, f'{path}/{key}')
            else:
                raise ValueError(f'Path {path} is not valid.')