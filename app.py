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
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipe = pipe.to(device)
pipe.enable_xformers_memory_efficient_attention()
torch.cuda.empty_cache()


refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-0.9", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
refiner = refiner.to(device)
refiner.enable_xformers_memory_efficient_attention()
torch.cuda.empty_cache()

def genie (prompt, negative_prompt, scale, steps, seed):
     torch.cuda.empty_cache()
     generator = torch.Generator(device=device).manual_seed(seed)
     int_image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=steps, guidance_scale=scale, num_images_per_prompt=1, generator=generator).images[0]
     image = refiner(prompt=prompt, image=int_image).images[0]
     return image 


    
gr.Interface(fn=genie, inputs=[gr.Textbox(label='What you want the AI to generate. 77 Token Limit.'), gr.Textbox(label='What you Do Not want the AI to generate.'), gr.Slider(1, 15, 10), gr.Slider(25, maximum=50, value=25, step=1), gr.Slider(minimum=1, step=1, maximum=999999999999999999, randomize=True)], outputs='image', title="Stable Diffusion XL .9 CPU", description="SDXL .9 CPU. <b>WARNING:</b> Extremely Slow. 65s/Iteration. Expect 25-50mins an image for 25-50 iterations respectively.", article = "Code Monkey: <a href=\"https://huggingface.co/Manjushri\">Manjushri</a>").launch(debug=True, max_threads=80)
gr.Interface(fn=refiner, inputs='image', outputs='image').launch(debug=True)