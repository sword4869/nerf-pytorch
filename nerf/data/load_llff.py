import numpy as np
import os, imageio



            
        
        
        
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    '''
    从poses_bounds.npy中提取出 poses, bds, imgs

    :param basedir: 'nerf_lllf_data/fern', 包含20张图片, poses_bounds.npy
    :return poses, bds, imgs: 都是样本数量在最后一维的格式
    '''
    ###### 拆分poses_bounds.npy，最后一维是样本数量。
    # (N,17), 即(20,17)， where N is the number of input images
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    # (3, 5, 20): (20, 17)→(20, 15)→(20, 3, 5)→(3,5,20)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    # (2, 20)：(20, 2)→(2, 20)
    bds = poses_arr[:, -2:].transpose([1,0])

    # 缩放图片
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    # H,W,C
    sh = imageio.imread(img0).shape
    
    sfx = ''
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    # 判断创建缩放后的图片文件夹成功了吗
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    # 判断poses内的pose数量匹配图片的数量吗
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    # 第五列的HWF，给其赋值缩放后的图片的HW
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    # 第五列的Focal按系数缩放
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    # 读入缩放后的图片
    if not load_imgs:
        return poses, bds
    
    # 这段代码可以去掉了， 'Gamma correction is now not applied anymore by default'
    # def imread(f):
    #     if f.endswith('png'):
    #         return imageio.imread(f, ignoregamma=True)
    #     else:
    #         return imageio.imread(f)

    # [0, 1.0]    
    imgs = [imageio.imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    # HWCN, 第一个pose的最后一列HWF
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs

    
            
            
    

def normalize(x):
    # 向量的单位化
    # 向量除以自己的二范数，得到和这个向量方向相同的单位向量。
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    '''
    已知z和任一xy平面的向量up，得到x和y向量。
    :param up: RUB, 自然up对应y轴
    :param pos: 即t平移向量
    :return : c2w, 3行4列。
    '''
    vec2 = normalize(z)
    vec1_avg = up
    # 叉乘得到x轴方向向量
    vec0 = normalize(np.cross(vec1_avg, vec2))
    # 叉乘得到，校准后的y轴方向向量
    # 这里其实不需要再次单位化: 两个互向正交的单位向量的叉乘，结果必然还是单位向量。$‖u×v‖=‖u‖⋅‖v‖⋅\sin⁡\theta, where \theta=90°$
    vec1 = normalize(np.cross(vec2, vec0))
    # [x轴，y轴，z轴，平移]
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):
    '''
    求多个输入相机的平均位姿c2w(R和t)
    '''
    hwf = poses[0, :3, -1:]

    # 第一步：t可以直接取平均值
    # 对多个相机的中心进行求均值得到center。
    center = poses[:, :3, 3].mean(0)
    # 第二步：R的三个轴不能简单取平均值，因为这样不能保证三轴互相垂直。
    # 1. 对所有相机的Z轴求和再求单位向量（就像下面的up一样不求单位向量也行，反正传入viewmatrix后都会再求单位向量）
    vec2 = normalize(poses[:, :3, 2].sum(0))
    # 2. 对所有的相机的Y轴求和。
    up = poses[:, :3, 1].sum(0)
    # 3. 将vec2, up, 和center输入到起对齐轴功能的viewmatrix()后，就可以得到平均的相机位姿了。
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    '''
    :return : 一段螺旋式的相机轨迹的pose, 但是这个pose是c2w还是w2c?
    :param c2w: 平均位姿
    :param up: y轴
    :param rads: shape (3,)。什么意思?
    :param rots: 2。什么意思?
    :param N: 多少个渲染视角。
    '''
    render_poses = []
    # shape: (4,)
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        # 将相机坐标下的点，转换到世界坐标下
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
    


def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    # 1. 得到多个输入相机的平均位姿c2w
    c2w = poses_avg(poses)
    # 2. 让平均位姿c2w和各个位姿poses，附加上[0,0,0,1]，变成完整的外参形式
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    # 3. 中心化
    # c2w是一个(4, 4)的矩阵，而poses是一个(N, 4, 4)的矩阵，得到(N, 4, 4)
    poses = np.linalg.inv(c2w) @ poses
    # 返回(N, 3, 4)
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    '''
    360°的环绕渲染位姿
    '''
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds
    

def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    '''
    @return images: (N, H, W, C), np.float32, [0, 1.0]
    @return poses: (N, 3, 5), R/t/hwf, np.float32
    @return bds: colmap生成的depth values that bound the closest and farthest scene content from that point of view（是离相机远近）.
    @return render_poses: np.float32
    @return i_test
    '''

    poses, bds, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    # 从DRB方向变成RUB方向
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    # poses, bds, imgs变成(N,...)的样本在第一维度的格式
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
    # Rescale if bd_factor is provided
    # 这放缩有什么意义呢？
    # R旋转矩阵与放缩无关，自然不变；也就t平移要放缩。为什么HWF不变呢？
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc
    
    # 中心化相机位姿
    if recenter:
        poses = recenter_poses(poses)
        
    # 渲染测试时的位姿，是360°还是Look forward 的椭圆形
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        # 中心化后的平均位姿
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        # 平均y轴的单位向量
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        # 位姿前后深度变化，Backward的z轴
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        # 120个视角
        N_views = 120
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
        
    render_poses = np.array(render_poses).astype(np.float32)

    # 只看平移量，找出平均位姿 c2w 和位姿 poses 间的最小平移量的位姿，其下标为 i_test
    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test



