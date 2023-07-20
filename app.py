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

def genie (prompt, negative_prompt, height, width, scale, steps, seed, upscaler):
    torch.cuda.max_memory_allocated(device='cuda')
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, vae=vae)
    pipe = pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention()
    torch.cuda.empty_cache()
    generator = torch.Generator(device=device).manual_seed(seed)
    int_image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=steps, height=height, width=width, guidance_scale=scale, num_images_per_prompt=1, generator=generator).images
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
    return (image, image)
   
gr.Interface(fn=genie, inputs=[gr.Textbox(label='What you want the AI to generate.<b> 77 Token Limit. A Token is Any Word, Number, Symbol, or Punctuation. Everything Over 77 Will Be Truncated!</b>'), 
    gr.Textbox(label='What you Do Not want the AI to generate. <b>77 Token Limit</b>'), 
    gr.Slider(512, 1024, 768, step=128, label='Height'),
    gr.Slider(512, 1024, 768, step=128, label='Width'),
    gr.Slider(1, 15, 10, step=.25, label='Guidance Scale: How Closely the AI follows the Prompt'), 
    gr.Slider(25, maximum=100, value=50, step=25, label='Number of Iterations'), 
    gr.Slider(minimum=1, step=1, maximum=999999999999999999, randomize=True, label='Seed'),
    gr.Radio(['Yes', 'No'], label='Upscale?')], 
    outputs=['image', 'image'],
    title="Stable Diffusion XL 0.9 GPU", 
    description="SDXL 0.9 GPU. <br><br><b>WARNING: Capable of producing NSFW (Softcore) images.</b>", 
    article = "Code Monkey: <a href=\"https://huggingface.co/Manjushri\">Manjushri</a>").launch(debug=True, max_threads=80)
