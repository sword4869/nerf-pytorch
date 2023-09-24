
class PointSampler():

    def __init__(self, H, W, focal, n_sample, near, far):
        self.H, self.W = H, W
        i, j = torch.meshgrid(
            torch.linspace(0, W - 1, W).to(device),
            torch.linspace(0, H - 1, H).to(device))
        i, j = i.t(), j.t()
        self.dirs = torch.stack(
            [(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i)],
            dim=-1).to(device)  # [H, W, 3]

        t_vals = torch.linspace(0., 1.,
                                steps=n_sample).to(device)  # [n_sample]
        self.z_vals = near * (1 - t_vals) + far * (t_vals)  # [n_sample]
        self.z_vals_test = self.z_vals[None, :].expand(
            H * W, n_sample)  # [H*W, n_sample]

    def sample_test(self, c2w):  # c2w: [3, 4]
        rays_d = torch.sum(
            self.dirs.unsqueeze(dim=-2) * c2w[:3, :3], dim=-1).view(
                -1,
                3)  # [H*W, 3] # TODO-@mst: improve this non-intuitive impl.
        rays_o = c2w[:3, -1].expand(rays_d.shape)  # [H*W, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * self.z_vals_test[
            ..., :, None]  # [H*W, n_sample, 3]
        return pts.view(pts.shape[0], -1)  # [H*W, n_sample*3]

    def sample_test2(self, c2w):  # c2w: [3, 4]
        rays_d = torch.sum(
            self.dirs.unsqueeze(dim=-2) * c2w[:3, :3], dim=-1).view(
                -1,
                3)  # [H*W, 3] # TODO-@mst: improve this non-intuitive impl.
        rays_o = c2w[:3, -1].expand(rays_d.shape)  # [H*W, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * self.z_vals_test[
            ..., :, None]  # [H*W, n_sample, 3]
        return pts  # [..., n_sample, 3]

    def sample_train(self, rays_o, rays_d, perturb):
        z_vals = self.z_vals[None, :].expand(
            rays_o.shape[0], self.z_vals.shape[0])  # depth [n_ray, n_sample]
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand(z_vals.shape).to(device)  # [n_ray, n_sample]
            z_vals = lower + (upper - lower) * t_rand
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[
            ..., :, None]  # [n_ray, n_sample, DIM_DIR]
        return pts.view(pts.shape[0], -1)  # [n_ray, n_sample * DIM_DIR]

    def sample_train2(self, rays_o, rays_d, perturb):
        '''rays_o: [n_img, patch_h, patch_w, 3] for CNN-style. Keep this for back-compatibility, please use sample_train_cnnstyle'''
        z_vals = self.z_vals[None, None, None, :].expand(
            *rays_o.shape[:3],
            self.z_vals.shape[0])  # [n_img, patch_h, patch_w, n_sample]
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1]
                         )  # [n_img, patch_h, patch_w, n_sample]
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape[0]).to(device)  # [n_img]
            t_rand = t_rand[:, None, None, None].expand_as(
                z_vals)  # [n_img, patch_h, patch_w, n_sample]
            z_vals = lower + (
                upper - lower) * t_rand  # [n_img, patch_h, patch_w, n_sample]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[
            ..., :, None]  # [n_img, patch_h, patch_w, n_sample, 3]
        return pts

    def sample_train_cnnstyle(self, rays_o, rays_d, perturb):
        '''rays_o and rayd_d: [n_patch, 3, patch_h, patch_w]'''
        z_vals = self.z_vals[None, None, None, :].expand(
            *rays_o.shape[:3],
            self.z_vals.shape[0])  # [n_img, patch_h, patch_w, n_sample]
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1]
                         )  # [n_img, patch_h, patch_w, n_sample]
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape[0]).to(device)  # [n_img]
            t_rand = t_rand[:, None, None, None].expand_as(
                z_vals)  # [n_img, patch_h, patch_w, n_sample]
            z_vals = lower + (
                upper - lower) * t_rand  # [n_img, patch_h, patch_w, n_sample]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[
            ..., :, None]  # [n_img, patch_h, patch_w, n_sample, 3]
        return pts

    def sample_train_plucker(self, rays_o, rays_d):
        r"""Use Plucker coordinates as ray representation.
        Refer to: https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf
        """
        m = torch.cross(rays_o, rays_d, dim=-1)  # [n_ray, 3]
        pts = torch.cat([rays_d, m], dim=-1)  # [n_ray, 6]
        return pts

    def sample_test_plucker(self, c2w):  # c2w: [3, 4]
        r"""Use Plucker coordinates as ray representation.
        """
        rays_d = torch.sum(
            self.dirs.unsqueeze(dim=-2) * c2w[:3, :3], dim=-1).view(
                -1,
                3)  # [H*W, 3] # TODO-@mst: improve this non-intuitive impl.
        rays_o = c2w[:3, -1].expand(rays_d.shape)  # [H*W, 3]
        m = torch.cross(rays_o, rays_d, dim=-1)  # [H*W, 3]
        pts = torch.cat([rays_d, m], dim=-1)  # [H*W, 6]
        return pts
    


def raw2outputs(raw,
                z_vals,
                rays_d,
                raw_noise_std=0,
                white_bkgd=False,
                pytest=False,
                global_step=-1,
                print=print):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(
        -act_fn(raw) * dists)  # @mst: opacity

    dists = z_vals[..., 1:] - z_vals[..., :-1]  # dists for 'distances'
    dists = torch.cat(
        [dists,
         torch.Tensor([1e10]).to(device).expand(dists[..., :1].shape)],
        -1)  # [N_rays, N_samples]
    # @mst: 1e10 for infinite distance

    dists = dists * torch.norm(
        rays_d[..., None, :],
        dim=-1)  # @mst: direction vector needs normalization. why this * ?

    rgb = torch.sigmoid(
        raw[..., :3])  # [N_rays, N_samples, 3], RGB for each sampled point
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

    # print to check alpha
    if global_step % 100 == 0:
        for i_ray in range(0, alpha.shape[0], 100):
            logtmp = ['%.4f' % x for x in alpha[i_ray]]
            print('%4d: ' % i_ray + ' '.join(logtmp))

    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(
        torch.cat(
            [torch.ones((alpha.shape[0], 1)).to(device), 1. - alpha + 1e-10],
            -1), -1)[:, :-1]  # @mst: [N_rays, N_samples]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map).to(device),
                              depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([
            fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)
        ], 0)

    return ret


def run_network(inputs,
                viewdirs,
                fn,
                embed_fn,
                embeddirs_fn,
                netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(
        inputs, [-1, inputs.shape[-1]]
    )  # @mst: shape: torch.Size([65536, 3]), 65536=1024*64 (n_rays * n_sample_per_ray)
    embedded = embed_fn(inputs_flat)  # shape: [n_rays*n_sample_per_ray, 63]

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat,
                            list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

