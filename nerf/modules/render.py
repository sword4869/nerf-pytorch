import torch
from einops import repeat


class Render:
    def __init__(
        self,
        model_coarse,
        model_fine,
        near,
        far,
        N_samples,
        N_importance,
        perturb
    ) -> None:
        self.model_coarse = model_coarse
        self.model_fine = model_fine
        self.near = near
        self.far = far
        self.N_samples = N_samples
        self.N_importance = N_importance
        self.perturb = perturb
        pass

    def render_rays(self, rays_o, rays_d, just_coarse=False):
        '''
        Args:
            rays_o: (N_rays, 3)
            rays_d: (N_rays, 3)
        '''
        device = rays_o.device

        near = self.near * torch.ones_like(rays_d[..., :1], device=device)                 # (N_rays, 1)
        far = self.far * torch.ones_like(rays_d[..., :1], device=device)
        t_vals = torch.linspace(0., 1., steps=self.N_samples).to(device)                   # (N_samples)
        # 即 near + (far-near) * t_vals, 让其在 near 到 far 之间等距离采点
        z_vals = near * (1.-t_vals) + far * (t_vals)                            # (N_rays, N_samples)

        # get intervals between samples
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        if self.perturb > 0.:
            upper = torch.cat([z_vals_mid, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], z_vals_mid], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)
            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]      # [N_rays, N_samples, 3]

        rgb_coarse, depth_coarse, disp_coarse, acc_coarse, weights_coarse, alpha_coarse = self.model_coarse(
            pts, rays_d, z_vals, self.N_samples)
        if just_coarse:
            return rgb_coarse, depth_coarse, disp_coarse, acc_coarse, weights_coarse, alpha_coarse

        z_samples = self._sample_pdf(z_vals_mid, weights_coarse[..., 1:-1])
        z_samples = z_samples.detach()  # [N_rays, N_importance]

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        rgb_fine, depth_fine, disp_fine, acc_fine, weights_fine, alpha_fine = self.model_fine(
            pts, rays_d, z_vals, self.N_samples + self.N_importance)
        return rgb_coarse, rgb_fine, depth_coarse, depth_fine, disp_coarse, disp_fine, acc_coarse, acc_fine, weights_coarse, weights_fine, alpha_coarse, alpha_fine

    def _sample_pdf(self, bins, weights):
        N_importance = self.N_importance
        device = bins.device

        # Get pdf
        weights = weights + 1e-5  # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1).to(device)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]).to(device), cdf], -1)  # (batch, len(bins))

        # Take uniform samples
        if self.perturb == 0:
            u = torch.linspace(0., 1., steps=N_importance).to(device)
            u = u.expand(list(cdf.shape[:-1]) + [N_importance])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_importance]).to(device)

        # Invert CDF
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds-1).to(device), inds-1)
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds).to(device), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_importance, 2)

        # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[..., 1]-cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u-cdf_g[..., 0])/denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

        return samples
