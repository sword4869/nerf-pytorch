import os
import numpy as np
import imageio
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange


from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

############################ Create NeRF Part ########################################


def create_nerf(args, near, far):
    """Instantiate NeRF's MLP model.
    """
    # 位置编码 location，由3维变成63维度。
    # 63 = 1 * 3 + （2 * 10）* 3 = 原本xyz + （sin, cos 10次）* 3维度
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        # 位置编码 direction，由3维变成27维度。
        # 27 = 3 + （2 * 4）* 3 = 原本diretion的三维度表示 + （sin, cos 4次）* 3维度
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    # MODIFY
    output_ch = 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    # network_fn 就是 model
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        # tar 文件的列表
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'near' : near,
        'far' : far
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


############################ Run NeRF Part ########################################

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """
    Arguments:
        inputs: (N_rays, N_points_per_ray, 3)
        viewdirs: (N_rays, 3)
        fn: model
        embed_fn, embeddirs_fn: embedder's function
        netchunk: chunk for run network
    Prepares inputs and applies network 'fn'.
    1. embed `inputs` and `viewdirs` -> `embedded`
    2. batchify
    """
    # (N_rays * N_points_per_ray, 3)
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    # fn(embedded)  or ret(embedded)
    # (N_rays * N_points_per_ray, 4)
    outputs_flat = batchify(fn, netchunk)(embedded)
    # (N_rays, N_points_per_ray, 4)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


############################ Render Part ########################################

def render_path(hwf, K, chunk, render_poses, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    '''
    渲染多个poses, 可以用来制作视频。render()只能渲染单个pose。

    Args:
        - `savedir`: 保存 render_poses 的 rgb图片
        - `render_factor`: 渲染更小的图片
        - `render_kwargs`: 这里只是传递字典变量

    Return: Ndarry
        - `rgbs`
        - `disps`
    '''

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor
        # MODIFY
        K[:2, :3] /= render_factor

    rgbs = []
    disps = []

    for i, c2w in enumerate(tqdm(render_poses)):
        ret = render(H, W, K, chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgb = ret['rgb_fine'] if render_kwargs['N_importance'] > 0 else ret['rgb_coarse']
        disp = ret['disp_fine'] if render_kwargs['N_importance'] > 0 else ret['disp_coarse']
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if savedir is not None:
            rgbs8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgbs8)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, c2w_staticcam=None,
            ndc=True, near=0., far=1., use_viewdirs=False, 
            **kwargs): 
    """Render rays
    Args:
    - 不由kwargs解析的:
    H: int. Height of image in pixels.
    W: int. Width of image in pixels.
    K: intrincs parameter of pinhole camera.
    chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
    rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
    c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
    c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
    camera while using other c2w argument for viewing directions.

    - 由kwargs解析的:
    ndc: bool. If True, represent ray origin, direction in NDC coordinates.
    near: float or array of shape [batch_size]. Nearest distance for a ray.
    far: float or array of shape [batch_size]. Farthest distance for a ray.
    use_viewdirs: bool. If True, use viewing direction of a point in space in model.

    Returns: Tensor
        将 render_rays() 的各变量，返回特定的shape。
        render_path() 调用 render() 的则会返回图片格式 (H, W, C)；直接调用 render(batch_rays) 则会返回 (N_rand, C)
    

    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    # 决定返回的ret是图片格式 [H, W, C] 还是 光线数量 [N_rand, C]，前者用于渲染poses， 即render_path(render_poses or poses[i_test]), 后者用于训练 render(batch_rays)
    sh = rays_d.shape # [H, W, 3] or [ N_rand, 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch, [ H*W, 3 ]
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    # 此时的kwargs是不包含ndc, near, far, use_viewdirs的
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        # ( [ H, W ] or [ N_rand ] ) + ( [3] or [] )
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    return all_ret


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    # ret means abbreviation of return
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            # {'rgb': [rgb1, rgb2, ..., rgb_num_chunk]}, num_chunk 表示根据 chunk 划分的块数, rgb_i: [N_rays, 3]
            all_ret[k].append(ret[k])
    # {'rgb': rgbs}. rgbs: [num_chunk * N_rays, 3]
    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render_rays(
    ray_batch,
    network_fn,
    network_query_fn,
    N_samples,
    lindisp=False,
    perturb=0.,
    N_importance=0,
    network_fine=None,
    white_bkgd=False,
    raw_noise_std=0.,
):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
    Returns:
        { rgb_coarse, disp_coarse, acc_coarse, depth_coarse | rgb_fine, disp_fine, acc_fine, depth_fine, z_std }
        { coase                                             | fine: N_import>0                                 }
        z_std: [N_rays]. Standard deviation of distances along ray for each sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        # 即 near + (far-near) * t_vals, 让其在 near 到 far 之间等距离采点
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_coarse, depth_coarse, disp_coarse, acc_coarse, weights_coarse, alpha_coarse = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    if N_importance > 0:

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights_coarse[...,1:-1], N_importance, det=(perturb==0.))
        z_samples = z_samples.detach()  # [N_rays, N_importance]

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_fine, depth_fine, disp_fine, acc_fine, weights_fine, alpha_fine = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)
    ret = {
        'rgb_coarse': rgb_coarse,
        'depth_coarse': depth_coarse,
        'disp_coarse': disp_coarse,
        'acc_coarse': acc_coarse,
        'weights_coarse': weights_coarse,
        'alpha_coarse': alpha_coarse
    }
    if N_importance > 0:
        ret.update({
            'rgb_fine': rgb_fine,
            'depth_fine': depth_fine,
            'disp_fine': disp_fine,
            'acc_fine': acc_fine,
            'weights_fine': weights_fine,
            'alpha_fine': alpha_fine,
            'z_std': torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        })

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [N_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [N_rays, num_samples along ray]. Integration time.
        rays_d: [N_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [N_rays, 3]. Estimated RGB color of a ray.
        depth_map: [N_rays]. Estimated distance to object.
        disp_map: [N_rays]. Disparity map. Inverse of depth map. 1 / depth.
        acc_map: [N_rays]. Sum of weights along each ray.  Accumulated opacity along each ray.
        weights: [N_rays, num_samples]. Weights assigned to each sampled color.
        alpha: [N_rays, N_samples]. Sigma for every point every ray.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1] # [N_rays, N_samples]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    # MODIFY
    acc_map = torch.sum(weights, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.max(1e-10 * torch.ones_like(acc_map), acc_map))

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, depth_map, disp_map, acc_map, weights, alpha




def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=10000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()


    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args, near, far)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(hwf, K, args.chunk, render_poses, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        # [N, 2, H, W, 3]，ro和rd的3是指坐标向量
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        # [N, 3, H, W, 3]，ro和rd的3是指坐标向量，rgb的3是指颜色通道
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        # [N, H, W, 3, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        # 训练集 [N_train, H, W, 3, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [N_train*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = 200000 + 1
    print('-'*15)
    print(f'TRAIN: {len(i_train)} {i_train}')
    print(f'VAL: {len(i_val)} {i_val}')
    print(f'TEST: {len(i_test)} {i_test}')
    print('-'*15)

    # Summary writers
    writer = SummaryWriter()


    start = start + 1
    pbar = trange(start, N_iters)
    for i in pbar:
        pbar.set_description(f"[TRAIN] Iter: {i} ")

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = torch.Tensor(images[img_i]).to(device)
            pose = poses[img_i, :3, :4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, pose)  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    # shape: (2dH, 2dW)，即2dW个横坐标，2dH个纵坐标。
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        ret = render(H, W, K, args.chunk, rays=batch_rays, **render_kwargs_train)

        optimizer.zero_grad()
        img_loss_coarse = img2mse(ret['rgb_coarse'], target_s)
        # coarse和fine 的都是分开算的
        psnr_coarse = mse2psnr(img_loss_coarse)
        loss = img_loss_coarse 

        if args.N_importance > 0:
            img_loss_fine = img2mse(ret['rgb_fine'], target_s)
            loss = loss + img_loss_fine
            psnr_fine = mse2psnr(img_loss_fine)
        psnr = psnr_fine if args.N_importance > 0 else psnr_coarse
        

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (i / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        
        # print
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        # save ckpt
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            ckpt = {
                'global_step': i,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if args.N_importance > 0:
                ckpt['network_fine_state_dict'] = render_kwargs_train['network_fine'].state_dict()
            torch.save(ckpt, path)
            tqdm.write(f'Saved checkpoints at {path}\n')

        # save video (rgb and disparity) of render_poses
        if i%args.i_video==0:
            tqdm.write(f'render poses...')
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(hwf, K, args.chunk, render_poses, render_kwargs_test)
            moviebase = os.path.join(basedir, expname, 'render_poses_{:06d}_'.format(i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        # poses[i_test]
        if i%args.i_testset==0:
            tqdm.write(f'test poses...')
            testsavedir = os.path.join(basedir, expname, 'test_poses_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                render_path(hwf, K, args.chunk, poses[i_test], render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)

        # tensorboard
        writer.add_scalar('loss/train', loss, i)
        writer.add_scalar('psnr/train', psnr, i)
        if i%args.i_img==0:
            img_i = np.random.choice(i_val)
            target = torch.Tensor(images[img_i]).to(device)
            with torch.no_grad():
                ret = render(H, W, K, chunk=args.chunk, c2w=poses[img_i], **render_kwargs_test)
            img_loss_coarse = img2mse(ret['rgb_coarse'], target)
            # coarse和fine 的都是分开算的
            psnr_coarse = mse2psnr(img_loss_coarse)
            loss = img_loss_coarse 

            if args.N_importance > 0:
                img_loss_fine = img2mse(ret['rgb_fine'], target)
                loss = loss + img_loss_fine
                psnr_fine = mse2psnr(img_loss_fine)
            psnr = psnr_fine if args.N_importance > 0 else psnr_coarse
            writer.add_scalar('psnr/val', psnr, i)
            writer.add_scalar('loss/val', loss, i)

            writer.add_image('val/rgb_coarse', ret['rgb_coarse'], i, dataformats='HWC')
            writer.add_image('val/disp_coarse', ret['disp_coarse'], i, dataformats='HW')
            writer.add_image('val/acc_coarse', ret['acc_coarse'], i, dataformats='HW')
            writer.add_image('val/depth_coarse', ret['depth_coarse'], i, dataformats='HW')
            writer.add_image('val/target', target, i, dataformats='HWC')

            if args.N_importance > 0:
                writer.add_image('val/rgb_fine', ret['rgb_fine'], i, dataformats='HWC')
                writer.add_image('val/disp_fine', ret['disp_fine'], i, dataformats='HW')
                writer.add_image('val/acc_fine', ret['acc_fine'], i, dataformats='HW')
                writer.add_image('val/depth_fine', ret['depth_fine'], i, dataformats='HW')
                writer.add_image('val/z_std', ret['z_std'], i, dataformats='HW')


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
