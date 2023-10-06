from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding():
    def __init__(self, multires, include_input=True, log_sampling=True):
        if log_sampling:
            self.freq_bands = 2 ** torch.linspace(0, multires - 1, steps=multires)  # [multires]
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** (multires - 1), steps=multires)

        self.include_input = include_input
        self.embed_num = 2 * multires + 1 if include_input else 2 * multires

    def cal_outdim(self, input_dims):
        return input_dims * self.embed_num
    
    def __call__(self, x):
        '''
        原始的 Positional encoding for input features
        [x, sin(x), cos(x), sin(2x), cos(2x), ..., sin(2^(multires-1)x), cos(2^(multires-1)x)]
        '''
        for i, freq in enumerate(self.freq_bands.to(x.device)):
            if i == 0:
                y = torch.concat([torch.sin(freq * x), torch.cos(freq * x)], dim=-1)
            else:
                y = torch.concat([y, torch.sin(freq * x), torch.cos(freq * x)], dim=-1)

        if self.include_input:
            y = torch.cat([x, y], dim=-1)

        return y

    '''
    # 就多了个 2 pi, 效果反而不如不加 
    # ![](images/2pi_psnr.png), ![](images/2pi_loss.png) 好的那个是不加 2 pi 的，其他两个是下面两个加了 2 pi 的实现
    # [x, sin(2*pi*x), cos(2*pi*x), sin(2*2*pi*x), cos(2*2*pi*x), ..., sin(2^(multires-1)*2*pi*x), cos(2^(multires-1)*2*pi*x))]
    def __call__(self, x):
        for i, freq in enumerate(self.freq_bands.to(x.device)):
            if i == 0:
                y = torch.concat([torch.sin(freq * 2 * torch.pi * x), torch.cos(freq * 2 * torch.pi * x)], dim=-1)
            else:
                y = torch.concat([y, torch.sin(freq * 2 * torch.pi * x), torch.cos(freq * 2 * torch.pi * x)], dim=-1)

        if self.include_input:
            y = torch.cat([x, y], dim=-1)

        return y
    
    
    # 下面是 https://github.com/nerfstudio-project/nerfstudio/blob/c2f5e68d548b66114a3dab0ad7707906a47126c9/nerfstudio/field_components/encodings.py#L93-L187 的简化
    # 这种方式输出维度的排列顺序与上面的不同，但网络结果相同(loss, psnr)、时间效率也没有明显差别。
    def __call__(self, x):
        y = 2 * torch.pi * x        # 2 pi x
        y = y[..., None] * self.freq_bands.to(x.device)     # freq * 2 pi x, # [...(最后一个是"input_dim"), "multires"]
        y = y.view(*y.shape[:-2], -1)                       # [..., "input_dim" * "multires"]
        y = torch.sin(torch.cat([y, y + torch.pi / 2.0], dim=-1))

        if self.include_input:
            y = torch.cat([x, y], dim=-1)

        return y
    '''    

class TriplaneEncoding(nn.Module):
    """Learned triplane encoding

    The encoding at [i,j,k] is an n dimensional vector corresponding to the element-wise product of the
    three n dimensional vectors at plane_coeff[i,j], plane_coeff[i,k], and plane_coeff[j,k].

    This allows for marginally more expressivity than the TensorVMEncoding, and each component is self standing
    and symmetrical, unlike with VM decomposition where we needed one component with a vector along all the x, y, z
    directions for symmetry.

    This can be thought of as 3 planes of features perpendicular to the x, y, and z axes, respectively and intersecting
    at the origin, and the encoding being the element-wise product of the element at the projection of [i, j, k] on
    these planes.

    The use for this is in representing a tensor decomp of a 4D embedding tensor: (x, y, z, feature_size)

    This will return a tensor of shape (bs:..., num_components)

    Args:
        resolution: Resolution of grid.
        num_components: The number of scalar triplanes to use (ie: output feature size)
        init_scale: The scale of the initial values of the planes
        product: Whether to use the element-wise product of the planes or the sum
    """

    # plane_coef: Float[Tensor, "3 num_components resolution resolution"]

    def __init__(
        self,
        resolution: int = 32,
        num_components: int = 64,
        init_scale: float = 0.1,
        reduce: Literal["sum", "product"] = "sum",
    ) -> None:
        super().__init__(in_dim=3)

        self.resolution = resolution
        self.num_components = num_components
        self.init_scale = init_scale
        self.reduce = reduce

        self.plane_coef = nn.Parameter(
            self.init_scale * torch.randn((3, self.num_components, self.resolution, self.resolution))
        )

    def get_out_dim(self) -> int:
        return self.num_components

    def forward(self, in_tensor):
        """Sample features from this encoder. Expects in_tensor to be in range [0, resolution]
        :param in_tensor: Float[Tensor, "*bs 3"])
        :return Float[Tensor, "*bs num_components featuresize"]:: 
        """

        original_shape = in_tensor.shape
        in_tensor = in_tensor.reshape(-1, 3)

        plane_coord = torch.stack([in_tensor[..., [0, 1]], in_tensor[..., [0, 2]], in_tensor[..., [1, 2]]], dim=0)

        # Stop gradients from going to sampler
        plane_coord = plane_coord.detach().view(3, -1, 1, 2)
        plane_features = F.grid_sample(
            self.plane_coef, plane_coord, align_corners=True
        )  # [3, num_components, flattened_bs, 1]

        if self.reduce == "product":
            plane_features = plane_features.prod(0).squeeze(-1).T  # [flattened_bs, num_components]
        else:
            plane_features = plane_features.sum(0).squeeze(-1).T

        return plane_features.reshape(*original_shape[:-1], self.num_components)

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underlying feature grid

        Args:
            resolution: Target resolution.
        """
        plane_coef = F.interpolate(
            self.plane_coef.data, size=(resolution, resolution), mode="bilinear", align_corners=True
        )

        self.plane_coef = torch.nn.Parameter(plane_coef)
        self.resolution = resolution
