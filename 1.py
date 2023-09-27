from einops import repeat
import torch
import torch.nn.functional as F
import numpy as np
from nerf.data.utils import pose_spherical, get_rays


def eq_eps_all(a, b):
    return torch.all(torch.abs(a - b) <= 1e-2)

def neq_eps_count(a, b):
    return torch.sum((torch.abs(a - b) > 1e-2).float())

def compare_model_output():
    raw = torch.load('raw.pth', map_location='cpu') 
    rgb_raw = raw[...,:3]  # [N_rays, N_samples, 3]
    sigma_raw = raw[...,3]

    rgb = torch.load('rgb.pt', map_location='cpu') 
    sigma = torch.load('sigma.pt', map_location='cpu')[:,:,0] 

    print(eq_eps_all(rgb_raw, rgb))
    print(eq_eps_all(sigma_raw, sigma))
    print(neq_eps_count(rgb_raw, rgb))
    print(neq_eps_count(sigma_raw, sigma))
    
    for i in range(len(rgb)):
        a = rgb[i]
        b = rgb_raw[i]
        if not eq_eps_all(a, b):
            print(i)

def compare_model_embed():
    input_pts = torch.load('input_pts.pt', map_location='cpu') 
    input_pts_raw = torch.load('input_pts_raw.pt', map_location='cpu').reshape(input_pts.shape[0], 64, -1)

    input_views = torch.load('input_views.pt', map_location='cpu') 
    input_views_raw = torch.load('input_views_raw.pt', map_location='cpu').reshape(input_views.shape[0], 64, -1)

    print(eq_eps_all(input_pts, input_pts_raw))
    print(eq_eps_all(input_views, input_views_raw))

    
    for i in range(len(input_pts)):
        a = input_pts[i]
        b = input_pts_raw[i]
        if not eq_eps_all(a, b):
            print(i)
            break

def compare_model_input():
    pts = torch.load('pts.pth', map_location='cpu') 
    pts_raw = torch.load('pts_raw.pth', map_location='cpu') 

    viewdirs = torch.load('viewdirs.pth', map_location='cpu') 
    viewdirs_raw = torch.load('viewdirs_raw.pth', map_location='cpu')
    viewdirs_raw = repeat(viewdirs_raw, 'b c -> b n c', n=64)

    print(eq_eps_all(pts, pts_raw))
    print(eq_eps_all(viewdirs, viewdirs_raw))

def compare_rays():
    rays_o = torch.load('rays_o.pt', map_location='cpu') 
    rays_o_raw = torch.load('rays_o_raw.pt', map_location='cpu') 

    rays_d = torch.load('rays_d.pt', map_location='cpu') 
    rays_d_raw = torch.load('rays_d_raw.pt', map_location='cpu') 


    print(eq_eps_all(rays_o, rays_o_raw))
    print(eq_eps_all(rays_d, rays_d_raw))
# compare_rays()
compare_model_input()
compare_model_embed()
compare_model_output()