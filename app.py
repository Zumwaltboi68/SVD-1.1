import gradio as gr
import torch
import numpy as np
import modin.pandas as pd
from PIL import Image
from diffusers import DiffusionPipeline 


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    PYTORCH_CUDA_ALLOC_CONF={'max_split_size_mb': 6000}
    torch.cuda.max_memory_allocated(device=device)
    torch.cuda.empty_cache()
    
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to(device)
    torch.cuda.empty_cache()
    
    refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", use_safetensors=True, torch_dtype=torch.float16, variant="fp16")
    refiner.enable_xformers_memory_efficient_attention()
    refiner = refiner.to(device)
    torch.cuda.empty_cache()
    
    upscaler = DiffusionPipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, use_safetensors=True)
    upscaler.enable_xformers_memory_efficient_attention()
    upscaler = upscaler.to(device)
    torch.cuda.empty_cache()
else: 
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True)
    pipe = pipe.to(device)
    refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", use_safetensors=True)
    refiner = refiner.to(device)
    
def genie (prompt, negative_prompt, height, width, scale, steps, seed, upscaling):
    generator = torch.Generator(device=device).manual_seed(seed)
    int_image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=steps, height=height, width=width, guidance_scale=scale, num_images_per_prompt=1, generator=generator, output_type="latent").images
    if upscaling == 'Yes':
        image = refiner(prompt=prompt, image=int_image).images[0]
        upscaled = upscaler(prompt=prompt, negative_prompt=negative_prompt, image=image, num_inference_steps=5, guidance_scale=0).images[0]
        torch.cuda.empty_cache()
        return (image, upscaled)
    else:
        image = refiner(prompt=prompt, negative_prompt=negative_prompt, image=int_image).images[0]   
        torch.cuda.empty_cache()
    return (image, image)
   
gr.Interface(fn=genie, inputs=[gr.Textbox(label='What you want the AI to generate. 77 Token Limit. A Token is Any Word, Number, Symbol, or Punctuation. Everything Over 77 Will Be Truncated!'), 
    gr.Textbox(label='What you Do Not want the AI to generate. 77 Token Limit'), 
    gr.Slider(512, 1024, 768, step=128, label='Height'),
    gr.Slider(512, 1024, 768, step=128, label='Width'),
    gr.Slider(1, 15, 10, step=.25, label='Guidance Scale: How Closely the AI follows the Prompt'), 
    gr.Slider(25, maximum=100, value=50, step=25, label='Number of Iterations'), 
    gr.Slider(minimum=1, step=1, maximum=999999999999999999, randomize=True, label='Seed'),
    gr.Radio(['Yes', 'No'], label='Upscale?')], 
    outputs=['image', 'image'],
    title="Stable Diffusion XL 1.0 GPU", 
    description="SDXL 1.0 GPU. <br><br><b>WARNING: Capable of producing NSFW (Softcore) images.</b>", 
    article = "Code Monkey: <a href=\"https://huggingface.co/Manjushri\">Manjushri</a>").queue(concurrency_count=1).launch(debug=True, max_threads=80)
