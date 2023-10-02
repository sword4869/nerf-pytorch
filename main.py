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

def train(global_step):
    dataset_train_precrop = BlenderPrecropRayDataset(split='train', **config['data']['params'], **config['data']['train']['precrop'])
    dataloader_train_precrop = torch.utils.data.DataLoader(dataset_train_precrop, **config['data']['train']['dataloader_params'])
    print(f'* Dataset loaded. Train rays {len(dataset_train_precrop)}')
    
    dataset_train_without_precrop = BlenderPrecropRayDataset(split='train', **config['data']['params'])
    dataloader_train_without_precrop = torch.utils.data.DataLoader(dataset_train_without_precrop, **config['data']['train']['dataloader_params'])
    print(f'* Dataset loaded. Train rays {len(dataset_train_without_precrop)}')

    def loader(global_step, dataloader_train, precrop):
        print('* Train precrop: ', precrop)
        while True:
            pbar = tqdm(dataloader_train)
            for i, batch in enumerate(pbar):
                if precrop:
                    if global_step > config['setting']['precrop_iters']:
                        return global_step
                else:
                    if global_step > config['setting']['step_num']:
                        return global_step

                global_step += 1
                pbar.set_description(f'{global_step} // {step_num}')
                
                rgb_original = batch['rgb_original'].to(device)
                rays_o = batch['rays_o'].to(device)
                rays_d = batch['rays_d'].to(device)
                rgb_coarse, rgb_fine, depth_coarse, depth_fine, disp_coarse, disp_fine, acc_coarse, acc_fine, weights_coarse, weights_fine, alpha_coarse, alpha_fine = render.render_rays(
                    rays_o, rays_d)
                loss_coarse = F.mse_loss(rgb_coarse, rgb_original)
                loss_fine = F.mse_loss(rgb_fine, rgb_original)
                loss = loss_coarse + loss_fine
                pbar.set_postfix({
                    'loss': loss.item(),
                    'loss_coarse': loss_coarse.item(),
                    'loss_fine': loss_fine.item()
                })
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

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

    
    global_step = loader(global_step, dataloader_train_precrop, precrop=True)
    global_step = loader(global_step, dataloader_train_without_precrop, precrop=False)


@torch.no_grad()
def val(global_step):
    val_dir = f'{logdir}/val_{global_step}'
    os.makedirs(val_dir, exist_ok=True)

    H, W = dataset_val.H, dataset_val.W

    img_buffer_hub = ImgBufferHub(H, W)

    assert len(dataset_val) % (H * W) == 0
    N_img = len(dataset_val) // (H * W)
    img_indexs = np.random.choice(N_img, config['data']['val']['other_params']['count'])
    batch_size = config['data']['val']['other_params']['batch_size']
    for img_index in img_indexs:
        for i in range(img_index * H * W, (img_index + 1) * H * W, batch_size):
            batch = dataset_val[i: i + batch_size ]
            rgb_original = batch['rgb_original'].to(device)
            rays_o = batch['rays_o'].to(device)
            rays_d = batch['rays_d'].to(device)
            rgb_coarse, rgb_fine, depth_coarse, depth_fine, disp_coarse, disp_fine, acc_coarse, acc_fine, weights_coarse, weights_fine, alpha_coarse, alpha_fine = render.render_rays(
                rays_o, rays_d)

            img_buffer_hub.update(val_dir, rgb_original=rgb_original, rgb_coarse=rgb_coarse, rgb_fine=rgb_fine, disp_coarse=disp_coarse, disp_fine=disp_fine)

@torch.no_grad()
def test(global_step):
    test_dir = f'{logdir}/test_{global_step}'
    os.makedirs(test_dir, exist_ok=True)

    dataset_test = BlenderPrecropRayDataset(split='test', **config['data']['params'])
    dataloader_test = torch.utils.data.DataLoader(dataset_test, **config['data']['test']['dataloader_params'])
    print(f'* Dataset loaded. Test rays {len(dataset_test)}')

    H, W = dataset_test.H, dataset_test.W

    img_buffer_hub = ImgBufferHub(H, W)

    for i, batch in enumerate(tqdm(dataloader_test)):
        rgb_original = batch['rgb_original'].to(device)
        rays_o = batch['rays_o'].to(device)
        rays_d = batch['rays_d'].to(device)
        rgb_coarse, rgb_fine, depth_coarse, depth_fine, disp_coarse, disp_fine, acc_coarse, acc_fine, weights_coarse, weights_fine, alpha_coarse, alpha_fine = render.render_rays(
            rays_o, rays_d)

        img_buffer_hub.update(test_dir, rgb_original=rgb_original, rgb_coarse=rgb_coarse, rgb_fine=rgb_fine)


if __name__ == '__main__':
    config = OmegaConf.load('config/lego.yaml')

    device = torch.device(config['setting']['device'])
    logdir = config['setting']['logdir']
    os.makedirs(logdir, exist_ok=True)

    dataset_val = BlenderPrecropRayDataset(split='val', **config['data']['params'])

    model_coarse = Nerf(**config['model_coarse']['params']).to(device)
    model_fine = Nerf(**config['model_fine']['params']).to(device)
    grad_vars = list(model_coarse.parameters()) + list(model_fine.parameters())
    optimizer = torch.optim.Adam(params=grad_vars, lr=config['setting']['lr'], betas=(0.9, 0.999))

    global_step = 0
    step_num = config['setting']['step_num']
    if config['setting']['ckpt'] is not None and config['setting']['ckpt'] != 'None':
        ckpt = torch.load(config['setting']['ckpt'])
        global_step = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        model_coarse.load_state_dict(ckpt['network_fn_state_dict'])
        model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        print(f'* Load ckpt from {config["setting"]["ckpt"]}')
        print(f'* global_step: {global_step}')

    render = Render(model_coarse, model_fine, **config['render']['params'])
    # train(global_step)
    # test(global_step)