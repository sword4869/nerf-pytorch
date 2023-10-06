import json
import os
from re import T

import numpy as np
from skimage import io
from skimage.transform import rescale
from sympy import N
import torch
from torch.utils.data import Dataset

from nerf.data.utils import pose_spherical, get_rays


class BlenderImageDataset(Dataset):
    def __init__(
        self,
        split,
        basedir,
        factor=1,
        skip=1,
        white_bkgd=True,
    ):
        self._load_blender_data(split, basedir, factor, skip, white_bkgd)

    def __len__(self):
        return len(self.rgb_original)

    def __getitem__(self, idx):
        return {
            'img': self.imgs[idx],
            'pose': self.poses[idx],
        }

    def _load_blender_data(self, split, basedir, factor, skip, white_bkgd):
        '''
        self.imgs: [N, H, W, C]
        self.poses: [N, 4, 4]
        [H, W, focal]
        '''

        self.imgs = []
        self.poses = []

        meta = {}
        with open(os.path.join(basedir, 'transforms_{}.json'.format(split)), 'r') as fp:
            meta = json.load(fp)

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            self.imgs.append(io.imread(fname) / 255.)
            self.poses.append(np.array(frame['transform_matrix']))

        self.H, self.W = self.imgs[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        self.focal = .5 * self.W / np.tan(.5 * camera_angle_x)

        if factor != 1:
            self.H = self.H // factor
            self.W = self.W // factor
            self.focal = self.focal / factor

            self.imgs_factor = []
            for img in self.imgs:
                self.imgs_factor.append(rescale(img, 1.0/factor, anti_aliasing=True, channel_axis=2))
            self.imgs = self.imgs_factor

        self.K = np.array([
            [self.focal, 0, 0.5 * self.W],
            [0, self.focal, 0.5 * self.H],
            [0, 0, 1]
        ])

        self.imgs = np.array(self.imgs).astype(np.float32)
        self.poses = np.array(self.poses).astype(np.float32)

        if white_bkgd:
            self.imgs = self.imgs[..., :3] * self.imgs[..., -1:] + (1. - self.imgs[..., -1:])
        else:
            self.imgs = self.imgs[..., :3]

    def _get_render_pose(self):
        # [-180., -171., ...,  171.] 每隔9度，共40个角度。
        render_poses = [pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40+1)[:-1]]
        render_poses = np.stack(render_poses, 0)
        return render_poses


class BlenderPrecropRayDataset(BlenderImageDataset):
    def __init__(
        self,
        split,
        basedir,
        factor,
        skip,
        white_bkgd,
        precrop_frac=None,
    ):
        self._load_blender_data(split, basedir, factor, skip, white_bkgd)
        self._load_ray(precrop_frac)

    def __len__(self):
        return len(self.rgb_original)

    def __getitem__(self, idx):
        return {
            'rgb_original': self.rgb_original[idx],
            'rays_o': self.rays_o[idx],
            'rays_d': self.rays_d[idx],
        }

    def _load_ray(self, precrop_frac):
        rays_o = []
        rays_d = []
        rgb_original = []
        for i in range(len(self.imgs)):
            img = torch.FloatTensor(self.imgs[i])
            pose = torch.FloatTensor(self.poses[i][:3, :4])
            ray_o, ray_d = get_rays(self.H, self.W, self.K, pose)

            if precrop_frac is not None:
                dH = int(self.H//2 * precrop_frac)
                dW = int(self.W//2 * precrop_frac)
                # shape: (2dH, 2dW)，即2dW个横坐标，2dH个纵坐标。
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(self.H//2 - dH, self.H//2 + dH - 1, 2*dH),
                        torch.linspace(self.W//2 - dW, self.W//2 + dW - 1, 2*dW)
                    ), -1)
                coords = torch.reshape(coords, [-1, 2]).long()  # (H * W, 2)
                # 舍弃了 N_rand
                ray_rgb = img[coords[:, 0], coords[:, 1]]  # (H * W, 3)
                ray_o = ray_o[coords[:, 0], coords[:, 1]]  # (H * W, 3)
                ray_d = ray_d[coords[:, 0], coords[:, 1]]  # (H * W, 3)
            
            else:
                ray_rgb = img.reshape(-1, 3)
                ray_o = ray_o.reshape(-1, 3)
                ray_d = ray_d.reshape(-1, 3)

            rgb_original.append(ray_rgb)
            rays_o.append(ray_o)
            rays_d.append(ray_d)

        self.rgb_original = torch.concatenate(rgb_original, 0)                          # (N * H * W, 3)
        self.rays_o = torch.concatenate(rays_o, 0)
        self.rays_d = torch.concatenate(rays_d, 0)