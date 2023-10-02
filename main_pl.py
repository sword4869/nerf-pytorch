import os
from typing import Any, Optional
from sympy import limit
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import os
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
from nerf.modules.render import Render
from nerf.model.nerf import Nerf
from nerf.data.blender_dataset import BlenderPrecropRayDataset
from nerf.utils import ImgBufferHub
import numpy as np


class Model(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self._load_model()
        self.save_hyperparameters()
        self.img_buffer_hub = ImgBufferHub(dataset_val.H, dataset_val.W)


    def _load_model(self):
        self.model_coarse = Nerf(**config['model_coarse']['params'])
        self.model_fine = Nerf(**config['model_fine']['params'])
        # self.global_step = 0

        if self.config['setting']['ckpt'] is not None and self.config['setting']['ckpt'] != 'None':
            ckpt = torch.load(self.config['setting']['ckpt'])
            # self.global_step = ckpt['global_step']
            # print(f'* global_step: {self.global_step}')
            self.model_coarse.load_state_dict(ckpt['network_fn_state_dict'])
            self.model_fine.load_state_dict(ckpt['network_fine_state_dict'])
            print(f'* Load ckpt from {config["setting"]["ckpt"]}')

        self.render = Render(self.model_coarse, self.model_fine, **config['render']['params'])

        os.makedirs(config['setting']['logdir'], exist_ok=True)
        print(f'* Logdir: {config["setting"]["logdir"]}')


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['setting']['lr'], betas=(0.9, 0.999))
        if self.config['setting']['ckpt'] is not None and self.config['setting']['ckpt'] != 'None':
            ckpt = torch.load(self.config['setting']['ckpt'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return optimizer

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> int | None:
        if self.global_step > config['setting']['step_num']:
            return -1

    def training_step(self, batch, batch_idx):
        rgb_original = batch['rgb_original']
        rays_o = batch['rays_o']
        rays_d = batch['rays_d']
        rgb_coarse, rgb_fine, depth_coarse, depth_fine, disp_coarse, disp_fine, acc_coarse, acc_fine, weights_coarse, weights_fine, alpha_coarse, alpha_fine = self.render.render_rays(
            rays_o, rays_d)
        loss_coarse = F.mse_loss(rgb_coarse, rgb_original)
        loss_fine = F.mse_loss(rgb_fine, rgb_original)
        loss = loss_coarse + loss_fine
        self.log_dict({
            'train_loss': loss.item(),
            'train_loss_coarse': loss_coarse.item(),
            'train_loss_fine': loss_fine.item()
        }, sync_dist=True, prog_bar=True)   
        return loss

        if global_step % config['setting']['i_val'] == 0:
            val(global_step)

        if global_step % config['setting']['i_ckpt'] == 0:
            ckpt = {
                'global_step': global_step,
                'optimizer_state_dict': optimizer.state_dict(),
                'network_fn_state_dict': render.model_coarse.state_dict(),
                'network_fine_state_dict': render.model_fine.state_dict()
            }
            torch.save(ckpt, f'{global_step}.pt')

    def on_validation_batch_start(self, batch: Any, batch_idx: int) -> int | None:
        self.val_dir = f"{self.config['setting']['logdir']}/val_{self.global_step}"
        os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_idx):
        rgb_original = batch['rgb_original']
        rays_o = batch['rays_o']
        rays_d = batch['rays_d']
        rgb_coarse, rgb_fine, depth_coarse, depth_fine, disp_coarse, disp_fine, acc_coarse, acc_fine, weights_coarse, weights_fine, alpha_coarse, alpha_fine = self.render.render_rays(
            rays_o, rays_d)
        loss_coarse = F.mse_loss(rgb_coarse, rgb_original)
        loss_fine = F.mse_loss(rgb_fine, rgb_original)
        loss = loss_coarse + loss_fine
        self.log_dict({
            'val_loss': loss.item(),
            'val_loss_coarse': loss_coarse.item(),
            'val_loss_fine': loss_fine.item()
        }, sync_dist=True, prog_bar=True)
        self.img_buffer_hub.update(self.val_dir, rgb_original=rgb_original, rgb_coarse=rgb_coarse, rgb_fine=rgb_fine, disp_coarse=disp_coarse, disp_fine=disp_fine)


if __name__ == '__main__':
    config = OmegaConf.load('config/lego_pl.yaml')

    dataset_train_precrop = BlenderPrecropRayDataset(split='train', **config['data']['params'], **config['data']['train']['precrop'])
    dataloader_train_precrop = torch.utils.data.DataLoader(dataset_train_precrop, **config['data']['train']['dataloader_params'])
    print(f'* Dataset loaded. Train rays {len(dataset_train_precrop)}')

    dataset_train_without_precrop = BlenderPrecropRayDataset(split='train', **config['data']['params'])
    dataloader_train_without_precrop = torch.utils.data.DataLoader(
        dataset_train_without_precrop, **config['data']['train']['dataloader_params'])
    print(f'* Dataset loaded. Train rays {len(dataset_train_without_precrop)}')

    dataset_val = BlenderPrecropRayDataset(split='val', **config['data']['params'])
    dataloader_val = torch.utils.data.DataLoader(dataset_val, **config['data']['val']['dataloader_params'])
    print(f'* Dataset loaded. Val rays {len(dataset_val)}')

    model = Model(config)
    limit_val_batches = dataset_val.H * dataset_val.W / config['data']['val']['dataloader_params']['batch_size'] * config['data']['val']['count'] 
    early_stopping = EarlyStopping(monitor="val_loss", verbose=True)
    trainer = pl.Trainer(
        limit_val_batches=limit_val_batches,
        devices=[1],
        callbacks=[early_stopping],
    )
    # trian the model
    trainer.fit(model=model, train_dataloaders=dataloader_train_without_precrop, val_dataloaders=dataloader_val)

    # test the model
    # dataset_test = BlenderPrecropRayDataset(split='test', **config['data']['params'])
    # dataloader_test = torch.utils.data.DataLoader(dataset_test, **config['data']['test']['dataloader_params'])
    # print(f'* Dataset loaded. Test rays {len(dataset_test)}')
    # trainer.test(model=model, dataloaders=test_loader)
