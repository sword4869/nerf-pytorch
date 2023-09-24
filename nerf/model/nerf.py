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
        y = x[..., None] * self.freq_bands                      # [n_ray, dim_pts, 1] * [multires] -> [n_ray, dim_pts, multires]
        y = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)     # [n_ray, dim_pts, 2L]
        if self.include_input:
            y = torch.cat([x.unsqueeze(dim=-1), y], dim=-1)     # [n_ray, dim_pts, 2L+1]

        return y.reshape(y.shape[0], -1)   # [n_ray, dim_pts*(2L+1)], example: 48*21=1008

    def cal_outdim(self, input_dims):
        return input_dims * self.embed_num


class NeRF(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        input_ch_views=3,
        output_ch=4,
        skips=[4],
        use_viewdirs=False
    ):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)]
        )

        # Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs
