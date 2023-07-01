# What is the difference between the origin code and forked code

## tensorboard

add tensorboard for
- loss of iteration for train
- configargparse 的 `i_img`  启用, show the img of rgb, disp, acc


## outputch

<https://github.com/yenchenlin/nerf-pytorch/issues/22>

> yenchenlin `output_ch = 5 if args.N_importance > 0 else 4` should be 4, but he didn't modiify the code.

`output_ch = 4` 只在不使用方向时`use_viewdirs=False`有效果，正常使用方向时就是4. 所以使用不使用 `use_viewdirs` 都是4.

## nan of disp


## render_factor of render_path

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

## none state_dict of ckpt

<https://github.com/yenchenlin/nerf-pytorch/pull/69>

在保存ckpt时，应该考虑到不使用精细模型的问题。