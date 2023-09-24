import os
import numpy as np
import imageio
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from run_nerf_helpers import *
from nerf.data.blender_dataset import BlenderDataset

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda")


def train():

    # Load data
    dataset = BlenderDataset(split='train', half_res=False, skip=1, basedir='data/nerf_synthetic/lego')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    print('Loaded', len(dataset), 'images')

    focal = dataset.focal
    H, W = dataset.H, dataset.W
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])

    model = NeRF()
    model_fine = NeRF()

    grad_vars = [model.parameters(), model_fine.parameters()]

    optimizer = torch.optim.Adam(params=grad_vars, lr=lrate, betas=(0.9, 0.999))

    