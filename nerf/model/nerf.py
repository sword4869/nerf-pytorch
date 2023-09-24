import torch
from torch import nn
import torch.nn.functional as F


class PositionalEmbedder():
    def __init__(self, multires, include_input=True, log_sampling=True):
        if log_sampling:
            self.freq_bands = 2 ** torch.linspace(0, multires - 1, steps=multires)  # [multires]
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** (multires - 1), steps=multires)

        self.include_input = include_input
        self.embed_num = 2 * multires + 1 if include_input else 2 * multires

    def __call__(self, x):
        y = x[..., None] * self.freq_bands                      # [..., dim_pts, 1] * [multires] -> [..., dim_pts, multires]
        y = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)     # [..., dim_pts, 2 * multires]
        if self.include_input:
            y = torch.cat([x.unsqueeze(dim=-1), y], dim=-1)     # [..., dim_pts, 2 * multires + 1]

        return y.reshape(y.shape[0], -1)   # [..., dim_pts * (2 * multires + 1)]

    def cal_outdim(self, input_dims):
        return input_dims * self.embed_num


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

        # Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W // 2, 3)


    def forward(self, pts, viewdirs):
        # (N_rand, N_samples, 3), (N_rand, 3)
        input_pts, input_views = self.pts_embedder(pts), self.viewdirs_embedder(viewdirs)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)    # (N_rand, N_samples, 1)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], -1)

        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)        # (N_rand, N_samples, 3)
        outputs = torch.cat([rgb, alpha], -1)
        return outputs