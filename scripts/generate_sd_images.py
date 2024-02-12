"""
The script generates images using stable diffusion

"""
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

cache_dir="/storage2/guangrun/qijia_3d_model/huggingface/"
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, cache_dir=cache_dir)
pipe = pipe.to('cuda')

positive_prompt = 'a photo of a plane, 8k, 4k'
negative_prompt = '3d, cartoon, anime, (deformed eyes, nose, ears, nose), bad anatomy, ugly'
image = pipe(prompt=positive_prompt, negative_prompt=negative_prompt, guidance_scale=5.0, num_images_per_prompt=10).images
for i in range(len(image)):
    image[i].resize((256, 256)).save(f'test_{i}.png')
