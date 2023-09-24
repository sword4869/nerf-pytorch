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


class BlenderDataset(Dataset):
    def __init__(
        self,
        split,
        basedir,
        half_res=False,
        skip=1,
        precrop_frac=0.5,
        N_rand=4096
    ):
        self.split = split
        if self.split == 'train':
            skip = 1
        self._load_blender_data(basedir, half_res, skip)
        self._load_ray_batch()
        self.precrop = False
        self.precrop_frac = precrop_frac
        self.N_rand = N_rand

    def __len__(self):
        return len(self.self.imgs)

    def __getitem__(self, idx):
        img = torch.FloatTensor(self.imgs[idx])
        pose = torch.FloatTensor(self.poses[idx][:3, :4])
        rays_o, rays_d = get_rays(self.H, self.W, self.focal, pose)

        if self.precrop:
            dH = int(self.H//2 * self.precrop_frac)
            dW = int(self.W//2 * self.precrop_frac)
            # shape: (2dH, 2dW)，即2dW个横坐标，2dH个纵坐标。
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(self.H//2 - dH, self.H//2 + dH - 1, 2*dH),
                    torch.linspace(self.W//2 - dW, self.W//2 + dW - 1, 2*dW)
                ), -1)
        else:
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, self.H-1, self.H),
                    torch.linspace(0, self.W-1, self.W)
                ), -1)  # (H, W, 2)

        coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
        select_inds = np.random.choice(coords.shape[0], size=[self.N_rand], replace=False)  # (N_rand,)
        select_coords = coords[select_inds].long()  # (N_rand, 2)
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        batch_rays = torch.stack([rays_o, rays_d], 0)
        img = img[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        return img, batch_rays, pose

    def _load_blender_data(
        self,
        basedir,
        factor=1,
        skip=1,
        white_bkgd=False
    ):
        '''
        self.imgs: [N, H, W, C]
        self.poses: [N, 4, 4]
        [H, W, focal]
        '''

        self.imgs = []
        self.poses = []
        counts = [0]

        meta = {}
        with open(os.path.join(basedir, 'transforms_{}.json'.format(self.split)), 'r') as fp:
            meta = json.load(fp)

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            self.imgs.append(io.imread(fname)[:, :, :3] / 255.)
            self.poses.append(np.array(frame['transform_matrix']))

        i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

        self.H, self.W = self.imgs[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        self.focal = .5 * self.W / np.tan(.5 * camera_angle_x)

        if factor != 1:
            self.H = self.H // factor
            self.W = self.W // factor
            focal = focal / factor

            self.imgs_factor = []
            for img in self.imgs:
                self.imgs_factor.append(rescale(img, 1.0/factor, anti_aliasing=True, channel_axis=2))
            self.imgs = self.imgs_factor

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


class BlenderPrecropDataset(Dataset):
    def __init__(
        self,
        split,
        basedir,
        factor=1,
        skip=1,
        white_bkgd=True,
        precrop_frac=0.5,
        N_rand=4096,
        near=2.0,
        far=6.0,
        N_samples=64,
        perturb=0.0
    ):
        if split == 'train':
            skip = 1
        elif split == 'val':
            pass
        elif split == 'test':
            perturb = 0

        self._load_blender_data(basedir, factor, skip, white_bkgd)
        self._load_ray_batch(precrop_frac, N_rand, near, far, N_samples, perturb)

    def __len__(self):
        return len(self.rays_o)

    def __getitem__(self, idx):
        return {
            'rays_rgb': self.rays_rgb[idx], 
            'pts': self.pts[idx], 
            'viewdirs': self.viewdirs[idx]
        }

    def _load_blender_data(self, basedir, factor, skip, white_bkgd):
        '''
        self.imgs: [N, H, W, C]
        self.poses: [N, 4, 4]
        [H, W, focal]
        '''

        self.imgs = []
        self.poses = []
        counts = [0]

        meta = {}
        with open(os.path.join(basedir, 'transforms_{}.json'.format(self.split)), 'r') as fp:
            meta = json.load(fp)

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            self.imgs.append(io.imread(fname)[:, :, :3] / 255.)
            self.poses.append(np.array(frame['transform_matrix']))

        i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

        self.H, self.W = self.imgs[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        self.focal = .5 * self.W / np.tan(.5 * camera_angle_x)

        if factor != 1:
            self.H = self.H // factor
            self.W = self.W // factor
            focal = focal / factor

            self.imgs_factor = []
            for img in self.imgs:
                self.imgs_factor.append(rescale(img, 1.0/factor, anti_aliasing=True, channel_axis=2))
            self.imgs = self.imgs_factor

        self.imgs = np.array(self.imgs).astype(np.float32)
        self.poses = np.array(self.poses).astype(np.float32)

        if white_bkgd:
            self.imgs = self.imgs[..., :3] * self.imgs[..., -1:] + (1. - self.imgs[..., -1:])
        else:
            self.imgs = self.imgs[..., :3]

    def _load_ray_batch(self, precrop_frac, N_rand, near, far, N_samples, perturb):
        rays_o = []
        rays_d = []
        rays_rgb = []
        for i in len(self.imgs):
            img = torch.FloatTensor(self.imgs[i])
            pose = torch.FloatTensor(self.poses[i][:3, :4])
            ray_o, ray_d = get_rays(self.H, self.W, self.focal, pose)

            dH = int(self.H//2 * precrop_frac)
            dW = int(self.W//2 * precrop_frac)
            # shape: (2dH, 2dW)，即2dW个横坐标，2dH个纵坐标。
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(self.H//2 - dH, self.H//2 + dH - 1, 2*dH),
                    torch.linspace(self.W//2 - dW, self.W//2 + dW - 1, 2*dW)
                ), -1)
            coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            ray_rgb = img[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            ray_o = ray_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            ray_d = ray_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_rgb.append(ray_rgb)
            rays_o.append(ray_o)
            rays_d.append(ray_d)

        self.rays_rgb = torch.concatenate(rays_rgb, 0)                          # (N * N_rand, 3)
        rays_o = torch.concatenate(rays_o, 0)
        rays_d = torch.concatenate(rays_d, 0)
        self.viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        self.viewdirs = self.viewdirs.expand(rays_d.shape[0], N_samples, 3)     # (N * N_rand, N_samples, 3)
        near = near * torch.ones_like(rays_d[..., :1])                          # (N * N_rand, 1)
        far = far * torch.ones_like(rays_d[..., :1])
        t_vals = torch.linspace(0., 1., steps=N_samples)                        # (N_samples)
        # 即 near + (far-near) * t_vals, 让其在 near 到 far 之间等距离采点
        z_vals = near * (1.-t_vals) + far * (t_vals)                            # (N * N_rand, N_samples)

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)
            z_vals = lower + (upper - lower) * t_rand

        self.pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]  # [N * N_rand, N_samples, 3]

    def _get_render_pose(self):
        # [-180., -171., ...,  171.] 每隔9度，共40个角度。
        render_poses = [pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40+1)[:-1]]
        render_poses = np.stack(render_poses, 0)
        return render_poses
