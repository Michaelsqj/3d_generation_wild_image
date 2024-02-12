import json
import torch
import os
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from tqdm import tqdm

def generate_images(ids: list, output_dir: str):
    num_images_per_prompt = 5

    offset2info_dict_path = '/storage2/guangrun/qijia_3d_model/imagenet/offset2info.json'
    with open(offset2info_dict_path, 'r') as f:
        offset2info_dict = json.load(f)

    cache_dir="/storage2/guangrun/qijia_3d_model/huggingface/"
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, cache_dir=cache_dir)
    pipe = pipe.to('cuda')

    for k, v in tqdm(offset2info_dict.items()):
        if v['idx'] not in ids:
            continue
        label = v['label']
        if not os.path.isdir(f'{output_dir}/{k}'):
            os.mkdir(f'{output_dir}/{k}')
        view_inst = ['', 'front', 'side', 'back']
        for n, view in enumerate(view_inst):
            positive_prompt = f"a photo of a {label}, {view}, 8k, 4k"
            negative_prompt = '3d, cartoon, anime, (deformed eyes, nose, ears, nose), bad anatomy, ugly'
            image = pipe(prompt=positive_prompt, negative_prompt=negative_prompt, guidance_scale=5.0, num_images_per_prompt=num_images_per_prompt).images
            for i in range(len(image)):
                # 04d means 4 digits, with leading zeros
                image[i].resize((256, 256)).save(f'{output_dir}/{k}/{k}_{(n*num_images_per_prompt+i):04d}.png')
        # positive_prompt = f"a photo of a {label}, 8k, 4k"
        # negative_prompt = '3d, cartoon, anime, (deformed eyes, nose, ears, nose), bad anatomy, ugly'
        
        # image = pipe(prompt=positive_prompt, negative_prompt=negative_prompt, guidance_scale=5.0, num_images_per_prompt=num_images_per_prompt).images
        # for i in range(len(image)):
        #     image[i].save(f'{output_dir}/{k}/{k}_{i}.png')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=str, default='0')
    parser.add_argument('--output_dir', type=str, default='/storage2/guangrun/qijia_3d_model/imagenet/sd_generated/')
    args = parser.parse_args()
    idx = list(range(int(args.idx.split(',')[0]), int(args.idx.split(',')[1])))
    generate_images(idx, args.output_dir)
