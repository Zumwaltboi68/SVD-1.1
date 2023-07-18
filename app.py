import gradio as gr
import torch
import numpy as np
import modin.pandas as pd
from PIL import Image
from diffusers import DiffusionPipeline 
from huggingface_hub import login
import os

login(token=os.environ.get('HF_KEY'))

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.cuda.max_memory_allocated(device=device)

def genie (prompt, negative_prompt, scale, steps, seed):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe = pipe.to(device)
    #pipe.enable_xformers_memory_efficient_attention()
    torch.cuda.empty_cache()
    generator = torch.Generator(device=device).manual_seed(seed)
    int_image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=steps, guidance_scale=scale, num_images_per_prompt=1, generator=generator, ).images
    torch.cuda.empty_cache()
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-0.9", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe = pipe.to(device)
    #pipe.enable_xformers_memory_efficient_attention()
    torch.cuda.empty_cache()
    image = pipe(prompt=prompt, image=int_image).images[0]
    return image
   
gr.Interface(fn=genie, inputs=[gr.Textbox(label='What you want the AI to generate. 77 Token Limit.'), 
    gr.Textbox(label='What you Do Not want the AI to generate.'), 
    gr.Slider(1, 15, 10), gr.Slider(25, maximum=100, value=50, step=1), 
    gr.Slider(minimum=1, step=1, maximum=999999999999999999, randomize=True)], 
    outputs='image', 
    title="Stable Diffusion XL 0.9 GPU", 
    description="SDXL 0.9 GPU. <b>WARNING:</b> Capable of producing NSFW images.", 
    article = "Code Monkey: <a href=\"https://huggingface.co/Manjushri\">Manjushri</a>").launch(debug=True, max_threads=80)
