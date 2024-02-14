"""
Test script for predicting depth 
"""
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, pipeline
import torch
import numpy as np
from PIL import Image
import requests




if __name__ == '__main__':
    img_path='/storage2/guangrun/qijia_3d_model/depth_anything/n02966687_0004.png'
    depth_path='/storage2/guangrun/qijia_3d_model/depth_anything/n02966687_0004_depth.png'
    cache_dir='/storage2/guangrun/qijia_3d_model/huggingface/'
    img = Image.open(img_path)
    image_processor = AutoImageProcessor.from_pretrained('LiheYoung/depth-anything-small-hf')
    model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
    inputs = image_processor(images=img, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    prediction = torch.nn.functional.interpolate(predicted_depth.unsqueeze(1),
                                                 size=img.size[::-1],
                                                 mode='bicubic',
                                                 align_corners=False)
    output = prediction.cpu().numpy().squeeze()
    formatted = ( output / np.max(output) *255 ).astype(np.uint8)
    depth = Image.fromarray(formatted)
    depth.save(depth_path)
    # pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
    # depth = pipe(img)["depth"]
    # depth = Image.fromarray(depth)
    # depth.save(depth_path)