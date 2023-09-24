import json
import os

import numpy as np
from skimage import io
from skimage.transform import rescale
from torch.utils.data import Dataset

from nerf.data.utils import pose_spherical


class BlenderDataset(Dataset):
    def __init__(
        self,
        split,
        basedir,
        half_res=False,
        skip=1
    ):
        self.split = split
        if self.split == 'train':
            skip = 1
        self._load_blender_data(basedir, half_res, skip)

    def __len__(self):
        return len(self.self.imgs)

    def __getitem__(self, idx):
        img = self.self.imgs[idx]
        pose = self.self.poses[idx]
        return img, pose

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