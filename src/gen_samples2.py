# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import mrcfile
import math


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = list(label_groups.keys())
        rnd.shuffle(label_order)
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=False)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=False)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=False)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    reload_modules: bool,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples2.py  \
            --outdir=/storage2/guangrun/qijia_3d_model/eg3d/htc/large_g_multi_view/00002-shapenet-imagenet_train_crop224_sz128-gpus4-batch32-gamma0.3/ \
            --network=/storage2/guangrun/qijia_3d_model/eg3d/htc/large_g_multi_view/00002-shapenet-imagenet_train_crop224_sz128-gpus4-batch32-gamma0.3/network-snapshot-004600.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].eval().to(device) # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if True:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    outdir = os.path.join(outdir, "gen_samples2")
    os.makedirs(outdir, exist_ok=True)

    # set up z and c
    batch_gpu = 32
    samples_per_cls = 2
    class_dim = G.c_dim
    cls_list = sorted([985,986,992,951,953,968,933,946,948,900,901,907,914,915,802,803,805,810,812,814,817,818,821,826,827,831,846,847,848,849,850,851,852,857,859,871,872,873,874,894,895,896,897,899,701,703,705,708,713,719,723,724,725,727,732,734,736,737,741,744,748,749,752,755,757,759,761,763,767,772,774,777,779,780,782,783,784,786,790,795])
    num_cls = len(cls_list)
    torch.manual_seed(0)
    grid_z = torch.randn([num_cls*samples_per_cls, G.z_dim], device=device).split(batch_gpu)
    grid_c = torch.zeros(num_cls, G.c_dim, device=device)
    grid_c[list(range(num_cls)), cls_list] = 1
    grid_c = grid_c.unsqueeze(1).repeat(1,samples_per_cls,1).reshape(-1, G.c_dim).split(batch_gpu)

    print("infer")
    out = []
    with torch.no_grad():
        for z, c in tqdm(zip(grid_z, grid_c)):
            tmp = G(z=z, c=c, noise_mode='const')
            out.append({'image':[img.cpu() for img in tmp['image']], 'image_raw':[img.cpu() for img in tmp['image_raw']], 'image_depth':[img.cpu() for img in tmp['image_depth']]})
    num_view = 6
    num_cls_per_img = 32
    print(f"num_cls {num_cls}")
    for i in range(num_view):
        images = torch.cat([o['image'][i] for o in out]).numpy()
        images_raw = torch.cat([o['image_raw'][i] for o in out]).numpy()
        images_depth = -torch.cat([o['image_depth'][i] for o in out]).numpy()
        print(images.shape)
        for j in range(int(math.ceil(num_cls / num_cls_per_img))):
            print(f'cls-{j*num_cls_per_img}-{min(num_cls, (j+1)*num_cls_per_img)}-view-{i}')
            gw = min(num_cls, (j+1)*num_cls_per_img)-j*num_cls_per_img
            gh = samples_per_cls
            grid_size = (gh, gw)

            start = j*num_cls_per_img * samples_per_cls
            end = min(num_cls, (j+1)*num_cls_per_img) * samples_per_cls
            save_image_grid(images[start:end,...], os.path.join(outdir, f'cls-{j*num_cls_per_img}-{min(num_cls, (j+1)*num_cls_per_img)}-view-{i}.png'), drange=[-1,1], grid_size=grid_size)
            save_image_grid(images_raw[start:end,...], os.path.join(outdir, f'cls-{j*num_cls_per_img}-{min(num_cls, (j+1)*num_cls_per_img)}-view-{i}_raw.png'), drange=[-1,1], grid_size=grid_size)
            save_image_grid(images_depth[start:end,...], os.path.join(outdir, f'cls-{j*num_cls_per_img}-{min(num_cls, (j+1)*num_cls_per_img)}-view-{i}_depth.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)



    # batch_size = 1
    # for class_idx in [16]:
    #     # Generate images.
    #     for seed_idx, seed in enumerate(seeds):
    #         # print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
    #         z = torch.from_numpy(np.random.RandomState(seed).randn(batch_size, G.z_dim)).to(device)

    #         # imgs = []
    #         # angle_p = -0.2
    #         # for angle_y, angle_p in [(.4, angle_p), (0, angle_p), (-.4, angle_p)]:
                
    #         class_condition = torch.zeros((batch_size,class_num), device=device)
    #         class_condition[:, class_idx] = 1
    #         conditioning_params = class_condition

    #         ws = G.mapping(z, class_condition, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
    #         out = G.synthesis(ws)

    #         for i in range(len(out['image'])):
    #             img = out['image'][i].cpu().numpy()
    #             drange=[-1,1]
    #             lo, hi = drange
    #             img = (img - lo) * (255 / (hi - lo))
    #             img = np.rint(img).clip(0, 255).astype(np.uint8)
    #             _N, C, H, W = img.shape
    #             img = img.transpose(0,2,3,1)[0,...]
    #             if C == 1:
    #                 PIL.Image.fromarray(img[:, :, 0], 'L').save(f'{outdir}/depth_seed{seed:04d}_view{i}.png')
    #             if C == 3:
    #                 PIL.Image.fromarray(img, 'RGB').save(f'{outdir}/rgb_seed{seed:04d}_view{i}.png')

    #             img = -out['image_depth'][i].cpu().numpy()
    #             lo, hi = np.min(img), np.max(img)
    #             img = (img - lo) * (255 / (hi - lo))
    #             img = np.rint(img).clip(0, 255).astype(np.uint8)
    #             _N, C, H, W = img.shape
    #             img = img.transpose(0,2,3,1)[0,...]
    #             if C == 1:
    #                 PIL.Image.fromarray(img[:, :, 0], 'L').save(f'{outdir}/depth_seed{seed:04d}_view{i}.png')
    #             if C == 3:
    #                 PIL.Image.fromarray(img, 'RGB').save(f'{outdir}/rgb_seed{seed:04d}_view{i}.png')
            
    #             # img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    #         #     imgs.append(img)
            
    #         grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
    #         save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
    #         grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
    #         grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
    #         # img = torch.cat(imgs, dim=2)
    #         out = [G_ema(z=z, c=c, noise_mode='const') for z, c in zip(grid_z, grid_c)]
    #         images = torch.cat([o['image'].cpu() for o in out]).numpy()
    #         images_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
    #         images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
    #         save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
    #         save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_raw.png'), drange=[-1,1], grid_size=grid_size)
    #         save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_depth.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)




            # if shapes:
            #     # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
            #     max_batch=1000000

            #     samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
            #     samples = samples.to(z.device)
            #     sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
            #     transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
            #     transformed_ray_directions_expanded[..., -1] = -1

            #     head = 0
            #     with tqdm(total = samples.shape[1]) as pbar:
            #         with torch.no_grad():
            #             while head < samples.shape[1]:
            #                 torch.manual_seed(0)
            #                 sigma = G.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')['sigma']
            #                 sigmas[:, head:head+max_batch] = sigma
            #                 head += max_batch
            #                 pbar.update(max_batch)

            #     sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
            #     sigmas = np.flip(sigmas, 0)

            #     # Trim the border of the extracted cube
            #     pad = int(30 * shape_res / 256)
            #     pad_value = -1000
            #     # sigmas[:pad] = pad_value
            #     # sigmas[-pad:] = pad_value
            #     # sigmas[:, :pad] = pad_value
            #     # sigmas[:, -pad:] = pad_value
            #     # sigmas[:, :, :pad] = pad_value
            #     # sigmas[:, :, -pad:] = pad_value

            #     if shape_format == '.ply':
            #         from shape_utils import convert_sdf_samples_to_ply
            #         convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, f'seed{seed:04d}.ply'), level=10)
            #     elif shape_format == '.mrc': # output mrc
            #         with mrcfile.new_mmap(os.path.join(outdir, f'seed{seed:04d}_class{class_idx:02d}.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
            #             mrc.data[:] = sigmas


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

