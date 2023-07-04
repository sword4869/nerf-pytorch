- [1. What is the difference between the origin code and forked code](#1-what-is-the-difference-between-the-origin-code-and-forked-code)
  - [1.1. tensorboard](#11-tensorboard)
  - [1.2. outputch](#12-outputch)
  - [1.3. nan of disp](#13-nan-of-disp)
  - [1.4. render\_factor of render\_path](#14-render_factor-of-render_path)
  - [1.5. none state\_dict of ckpt](#15-none-state_dict-of-ckpt)
  - [1.6. global\_step](#16-global_step)
  - [1.7. 输出名称](#17-输出名称)

---
# 1. What is the difference between the origin code and forked code

## 1.1. tensorboard

add tensorboard for
- loss of iteration for train
- configargparse 的 `i_img`  启用, show the img of rgb, disp, acc


## 1.2. outputch

<https://github.com/yenchenlin/nerf-pytorch/issues/22>

> yenchenlin `output_ch = 5 if args.N_importance > 0 else 4` should be 4, but he didn't modiify the code.

`output_ch = 4` 只在不使用方向时`use_viewdirs=False`有效果，正常使用方向时就是4. 所以使用不使用 `use_viewdirs` 都是4.

## 1.3. nan of disp

重用了 `acc_map`；`depth_map / acc_map`也需要防止除0.
```python
acc_map = torch.sum(weights, -1)
disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.max(1e-10 * torch.ones_like(acc_map), acc_map))
```

## 1.4. render_factor of render_path

<https://github.com/yenchenlin/nerf-pytorch/pull/119>

`K` 也要变化
```python
    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor
        # MODIFY
        K[:2, :3] /= render_factor
```

## 1.5. none state_dict of ckpt

<https://github.com/yenchenlin/nerf-pytorch/pull/69>

在保存ckpt时，应该考虑到不使用精细模型的问题。

## 1.6. global_step 

`global_step` 与 `i` 重复了.

另外，修改了进度条的样式，加了表示 `i` 的前缀。

## 1.7. 输出名称

- render_poses的视频：

  `fern_test_spiral_200000_rgb.mp4` -> `render_poses_200000_rgb.mp4`

  `fern_test_spiral_200000_disp.mp4` -> `render_poses_200000_disp.mp4`

- test_poses的图片

  `testset_010000` -> `test_poses_010000`
