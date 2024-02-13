import gradio as gr
import torch
import numpy as np
import modin.pandas as pd
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from huggingface_hub import login
import os


token = os.environ['HF_TOKEN']
login(token=token)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.max_memory_allocated(device=device)
torch.cuda.empty_cache()
pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid", use_safetensors=True)
#pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

pipe.enable_xformers_memory_efficient_attention()
pipe = pipe.to(device)
torch.cuda.empty_cache()



def genie(src_image):
    torch.cuda.max_memory_allocated(device=device)
    torch.cuda.empty_cache()
    
    frames = pipe(image=src_image).images[0]
    torch.cuda.empty_cache()
    return frames
    
gr.Interface(fn=genie, inputs=gr.Image(type="pil"), outputs=gr.Video()).launch(debug=True, max_threads=80)