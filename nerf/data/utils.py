import torch
import numpy as np
import os
from skimage import io
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import pickle


def trans_t(t):
    # the translation along the z-axis. It returns a 4x4 numpy array that represents a translation matrix.
    return np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, t],
         [0, 0, 0, 1]],
        dtype=np.float32
    )


def rot_phi(phi):
    # rotation of polar angle, rotation around the x-axis.
    # It returns a 4x4 numpy array that represents a rotation matrix.
    return np.array(
        [[1, 0, 0, 0],
         [0, np.cos(phi), -np.sin(phi), 0],
         [0, np.sin(phi), np.cos(phi), 0],
         [0, 0, 0, 1]],
        dtype=np.float32
    )


def rot_theta(th):
    # rotation of azimuthal angle, rotation around the y-axis.
    # It returns a 4x4 numpy array that represents a rotation matrix.
    return np.array(
        [[np.cos(th), 0, -np.sin(th), 0],
         [0, 1, 0, 0],
         [np.sin(th), 0, np.cos(th), 0],
         [0, 0, 0, 1]],
        dtype=np.float32
    )


def pose_spherical(theta, phi, radius):
    """
    Returns the camera-to-world matrix that represents the position and orientation of a camera in 3D space, given the
    spherical coordinates of the camera.

    Args:
        theta (float): The azimuthal angle (in degrees) of the camera.
        phi (float): The polar angle (in degrees) of the camera.
        radius (float): The distance between the camera and the origin.

    Returns:
        torch.Tensor: The 4x4 camera-to-world matrix.
    """

    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    coordinate = np.array(
        [[-1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]]
    )
    c2w = coordinate @ c2w
    return c2w


def visualize_3d(
    xyzs,
    cmaps,
    connect=False,
    lim=None
):
    fig = plt.figure()
    ax3d = plt.axes(projection='3d')
    for ix, item in enumerate(xyzs):
        x, y, z = item
        ax3d.scatter3D(x, y, z, cmap=cmaps[ix])
        if connect:
            ax3d.plot3D(x, y, z)
    ax3d.scatter3D(0, 0, 0, marker='d', color='red')
    label_fs, ticklabelsize = 14, 9
    ax3d.set_xlabel('X axis', fontsize=label_fs)
    ax3d.set_ylabel('Y axis', fontsize=label_fs)
    ax3d.set_zlabel('Z axis', fontsize=label_fs)
    ax3d.tick_params(axis='both', labelsize=ticklabelsize)
    plt.show()


def minify(basedir, factors=[], resolutions=[]):
    '''
    NOTE: 
    - No special needs: suitable for all platforms, using `skimage` to downscale instead of using special cli `mogrify`

    according to the `factor` or `resolution` to scale the images in `images` and save the result to the corresponding subdirectory `images_{}` or `images_{}x{}`.
    `factor` and `resolution` can be both used.

    :param basedir: the parent directory of `images` directory
    :param factors: HW同比例缩放几倍, list for many different factors, e.g. `images_4` and `images_8` for the `factors=[4, 8]`
    :param resolutions: HW缩放到某个尺寸, e.g. `resolutions=[(300, 400)]`, 新图片的HW=(300,400)
    '''

    # 根据其要求的文件夹，如果有一个不存在，就重新生成。否则，都在就直接return。
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[0], r[1]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    imgdir = os.path.join(basedir, 'images')
    # 供保存图片使用
    img_names = [f for f in sorted(os.listdir(imgdir)) if any([f.lower().endswith(ex) for ex in ['jpg', 'jpeg', 'png']])]

    # 读取图片
    imgs = [io.imread(os.path.join(imgdir, img_name))[:, :, :3] / 255. for img_name in img_names]

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
        else:
            name = 'images_{}x{}'.format(r[0], r[1])
        imgdir = os.path.join(basedir, name)

        if os.path.exists(imgdir):
            continue
        print('Minifying', r, basedir)

        os.makedirs(imgdir)

        for i in range(len(imgs)):
            if isinstance(r, int):
                imgs_down = rescale(imgs[i], 1.0/r, anti_aliasing=True, channel_axis=2)
            else:
                imgs_down = resize(imgs[i], (r[0], r[1]), anti_aliasing=True)
            imgs_down = (imgs_down * 255).astype(np.uint8)
            io.imsave(os.path.join(imgdir, img_names[i]), imgs_down)


# Ray helpers
def get_rays(H, W, K, c2w):
    # 因为 pytorch's meshgrid 的indexing不同，所以这三行只是在等效 np.meshgrid，直接看get_rays_np
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='xy')
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    '''
    给定一张图像的一个像素点，我们的目标是构造以相机中心为起始点，经过相机中心和像素点的射线。
    @param H,W :图像的宽、高
    @param K: 内参
    @param c2w : 外参，[3,4], `c2w[:3,-1]`最后一列需要是t
    @return rays_o: (H, W, 3). HxW个光线在世界坐标中的起点位置, 即HxW个一样的三个坐标。
    @return rays_d: (H, W, 3). HxW个光线dirs。
    '''
    # input: (W, H)， output: (H, W)，行坐标有W个，纵坐标有H个。
    i, j = np.meshgrid(np.linspace(0, W-1, W), np.linspace(0, H-1, H), indexing='xy')
    # 射线方向 + 不同的相机坐标系转化
    # (378, 504, 3)：相机坐标下，每个坐标对应三个值来表示射线方向
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # (378, 504, 1, 3) * (3, 3) = (378, 504, 3, 3)， 最后一个维度 (378, 504, 3)
    # 相当于 rays_d = (dirs[..., np.newaxis, :] @ c2w[:3,:3].T).transpose(0,1,3,2).squeeze(-1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

if __name__ == '__main__':
    render_poses = [pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40+1)[:-1]]
    render_poses = np.stack(render_poses, 0)
    cmaps = ['Greens']
    xyzs = [(render_poses[:, 0, 3], render_poses[:, 1, 3], render_poses[:, 2, 3])]
    visualize_3d(xyzs=xyzs, cmaps=cmaps)
