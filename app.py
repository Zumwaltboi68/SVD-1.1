import gradio as gr
import torch
import numpy as np
import modin.pandas as pd
from PIL import Image
from diffusers import DiffusionPipeline 
from huggingface_hub import login
import os
from diffusers.models import AutoencoderKL

login(token=os.environ.get('HF_KEY'))

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.max_memory_allocated(device='cuda')
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float16)
torch.cuda.empty_cache()

def genie (prompt, negative_prompt, scale, steps, seed, upscaler):
    torch.cuda.max_memory_allocated(device='cuda')
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, vae=vae)
    pipe = pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention()
    torch.cuda.empty_cache()
    generator = torch.Generator(device=device).manual_seed(seed)
    int_image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=steps, guidance_scale=scale, num_images_per_prompt=1, generator=generator).images
    torch.cuda.empty_cache()
    if upscaler == 'Yes':
        torch.cuda.max_memory_allocated(device='cuda')
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-0.9", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, vae=vae)
        pipe = pipe.to(device)
        pipe.enable_xformers_memory_efficient_attention()
        image = pipe(prompt=prompt, image=int_image).images[0]
        torch.cuda.empty_cache()
        torch.cuda.max_memory_allocated(device='cuda')
        pipe = DiffusionPipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, use_safetensors=True)
        pipe.to("cuda")
        pipe.enable_xformers_memory_efficient_attention()
        upscaled = pipe(prompt=prompt, negative_prompt=negative_prompt, image=image, num_inference_steps=5, guidance_scale=0).images[0]
        torch.cuda.empty_cache()
        return (image, upscaled)
    else:
        torch.cuda.empty_cache()
        torch.cuda.max_memory_allocated(device=device)
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-0.9", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, vae=vae)
        pipe = pipe.to(device)
        pipe.enable_xformers_memory_efficient_attention()
        image = pipe(prompt=prompt, image=int_image).images[0]
        torch.cuda.empty_cache()
    return image
   
gr.Interface(fn=genie, inputs=[gr.Textbox(label='What you want the AI to generate. 77 Token Limit.'), 
    gr.Textbox(label='What you Do Not want the AI to generate.'), 
    gr.Slider(1, 15, 10), gr.Slider(25, maximum=100, value=50, step=1), 
    gr.Slider(minimum=1, step=1, maximum=999999999999999999, randomize=True),
    gr.Radio(['Yes', 'No'], label='Upscale?')], 
    outputs=['image', 'image'],
    title="Stable Diffusion XL 0.9 GPU", 
    description="SDXL 0.9 GPU. <b>WARNING:</b> Capable of producing NSFW images.", 
    article = "Code Monkey: <a href=\"https://huggingface.co/Manjushri\">Manjushri</a>").launch(debug=True, max_threads=80)
