import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn

from .embedder import PositionalEmbedder


class NeRF(nn.Module):
    def __init__(
        self,
        D,
        W,
        input_ch,
        input_ch_views,
        skips,
        multires,
        multires_views,
        raw_noise_std,
        white_bkgd
    ):
        super().__init__()

        self.pts_embedder = PositionalEmbedder(multires)
        self.viewdirs_embedder = PositionalEmbedder(multires_views)
        input_ch = self.pts_embedder.cal_outdim(input_ch)
        input_ch_views = self.viewdirs_embedder.cal_outdim(input_ch_views)

        self.skips = skips

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)]
        )
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W // 2, 3)

        self.raw_noise_std = raw_noise_std
        self.white_bkgd = white_bkgd

    def forward(self, pts, rays_d, z_vals, N_samples):
        viewdirs_norm = torch.norm(rays_d, dim=-1, keepdim=True).to(rays_d.device)      # [N_rays, 1]
        viewdirs = rays_d / viewdirs_norm                                               # [N_rays, 3]
        viewdirs = repeat(viewdirs, 'b c -> b n c', n=N_samples)                        # [N_rays, N_samples, 3]
        input_pts, input_views = self.pts_embedder(pts), self.viewdirs_embedder(viewdirs)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        sigma = self.alpha_linear(h)    # (N_rays, N_samples, 1)
        sigma = F.relu(sigma)

        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], -1)
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)        # (N_rays, N_samples, 3)
        rgb = F.sigmoid(rgb)

        return self.raw2outputs(rgb, sigma, z_vals, viewdirs_norm)

    def raw2outputs(self, rgb, sigma, z_vals, viewdirs_norm):
        device = z_vals.device

        dists = z_vals[..., 1:] - z_vals[..., :-1]      # [N_rays, N_samples - 1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(device)], -1)  # [N_rays, N_samples]
        dists = dists * viewdirs_norm    # [N_rays, N_samples] * [N_rays, 1] = [N_rays, N_samples]

        if self.raw_noise_std > 0.:
            noise = torch.randn(sigma.shape) * self.raw_noise_std
            sigma += noise
        alpha = 1 - torch.exp(- sigma.squeeze(-1) * dists)      # [N_rays, N_samples] * [N_rays, N_samples] = [N_rays, N_samples]

        T = torch.cumprod(1. - alpha + 1e-10, -1)[..., :-1]   # [N_rays, N_samples - 1]
        T = torch.concat([torch.ones((T.shape[0], 1)).to(device), T], -1)  # [N_rays, N_samples]
        weights = alpha * T     # [N_rays, N_samples]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, N_samples, 1] * [N_rays, N_samples, 3] -> [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        # MODIFY
        acc_map = torch.sum(weights, -1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map, device=device),
                                depth_map / torch.max(1e-10 * torch.ones_like(acc_map, device=device), acc_map))

        if self.white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[..., None])

        return rgb_map, depth_map, disp_map, acc_map, weights, alpha
