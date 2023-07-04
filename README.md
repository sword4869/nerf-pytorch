- [1. NeRF-pytorch](#1-nerf-pytorch)
  - [1.1. Installation](#11-installation)
  - [1.2. Dataset](#12-dataset)
    - [1.2.1. example dataset](#121-example-dataset)
    - [1.2.2. More Datasets](#122-more-datasets)
  - [1.3. Train](#13-train)
  - [1.4. QuickStart](#14-quickstart)


---
# 1. NeRF-pytorch

[Paper](https://arxiv.org/abs/2003.08934)

[Project](http://www.matthewtancik.com/nerf) 

> code

[Tensorflow implementation](https://github.com/bmild/nerf)

[nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)

## 1.1. Installation

```bash
git clone git@github.com:sword4869/nerf-pytorch.git
cd nerf-pytorch
conda create -n nerf python=3.10 -y
conda activate nerf
pip install -r requirements.txt
```

The LLFF data loader requires ImageMagick.

You will also need the [LLFF code](http://github.com/fyusion/llff) (and COLMAP) set up to compute poses if you want to run on your own real data.
  

## 1.2. Dataset
### 1.2.1. example dataset
Download data for two example datasets: `nerf_llff_data` 的 `fern` 和 `nerf_synthetic` 的 `lego`

```bash
# this bash is so lower than directly downloading the following link.
bash download_example_data.sh
```

### 1.2.2. More Datasets

To play with other scenes presented in the paper, download the data [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). Place the downloaded dataset according to the following directory structure:
```
├── configs                                                                                                       
│   ├── ...                                                                                     
│                                                                                               
├── data                                                                                                                                                                                                       
│   ├── nerf_llff_data                                                                                                  
│   │   └── fern                                                                                                                             
│   │   └── flower  # downloaded llff dataset                                                                                  
│   │   └── horns   # downloaded llff dataset
|   |   └── ...
|   ├── nerf_synthetic
|   |   └── lego
|   |   └── ship    # downloaded synthetic dataset
|   |   └── ...
```
## 1.3. Train

> To train a low-res `lego` NeRF:
```bash
python run_nerf.py --config configs/lego.txt
```
After training for 100k iterations (~4 hours on a single 2080 Ti), you can find the following video at `logs/lego_test/lego_test_spiral_100000_rgb.mp4`.

![](https://user-images.githubusercontent.com/7057863/78473103-9353b300-7770-11ea-98ed-6ba2d877b62c.gif)



> To train a low-res `fern` NeRF:
```bash
python run_nerf.py --config configs/fern.txt
```
After training for 200k iterations (~8 hours on a single 2080 Ti), you can find the following video at `logs/fern_test/render_poses_200000_rgb.mp4` and `logs/fern_test/render_poses_200000_disp.mp4`

![](https://user-images.githubusercontent.com/7057863/78473081-58ea1600-7770-11ea-92ce-2bbf6a3f9add.gif)


## 1.4. QuickStart

You can download the pre-trained models [here](https://drive.google.com/drive/folders/1jIr8dkvefrQmv737fFm2isiT6tqpbTbv). Place the downloaded directory in `./logs` in order to test it later. See the following directory structure for an example:

```bash
├── logs 
│   ├── fern_test
│   ├── flower_test  # downloaded logs
│   ├── trex_test    # downloaded logs
```

```bash
python run_nerf.py --config configs/{DATASET}.txt --render_only
```