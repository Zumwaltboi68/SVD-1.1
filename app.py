import torch #needed only for GPU
from PIL import Image
from io import BytesIO
import numpy as np
from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionUpscalePipeline
import gradio as gr
import modin.pandas as pd

# load model for CPU or GPU

model_2x = "stabilityai/sd-x2-latent-upscaler"
model_4x = "stabilityai/stable-diffusion-x4-upscaler"
device = "cuda" if torch.cuda.is_available() else "cpu"
upscaler2x = StableDiffusionLatentUpscalePipeline.from_pretrained(model_2x, torch_dtype=torch.float16) if torch.cuda.is_available() else StableDiffusionLatentUpscalePipeline.from_pretrained(model_2x)
upscaler4x = StableDiffusionUpscalePipeline.from_pretrained(model_4x, torch_dtype=torch.float16, revision="fp16") if torch.cuda.is_available() else StableDiffusionUpscalePipeline.from_pretrained(model_4x)
upscaler2x = upscaler2x.to(device)
upscaler4x = upscaler4x.to(device)

#define interface 

def upscale(raw_img, model, prompt, negative_prompt, scale, steps, Seed):
    generator = torch.manual_seed(Seed)
    raw_img = Image.open(raw_img).convert("RGB")
    if model == "Upscaler 4x":
        low_res_img = raw_img.resize((128, 128))
        upscaled_image = upscaler4x(prompt=prompt, negative_prompt=negative_prompt, image=low_res_img, guidance_scale=scale, num_inference_steps=steps).images[0]
    else: 
        upscaled_image = upscaler2x(prompt=prompt, negative_prompt=negative_prompt, image=raw_img, guidance_scale=scale, num_inference_steps=steps).images[0]
    return upscaled_image
    
#launch interface
    
gr.Interface(fn=upscale, inputs=[
    gr.Image(type="filepath", label='Lower Resolution Image'), 
    gr.Radio(['Upscaler 2x','Upscaler 4x'], label="Models"),
    gr.Textbox(label="Optional: Enter a Prompt to Guide the AI's Enhancement, this can have an Img2Img Effect"), 
    gr.Textbox(label='Experimental: Influence What you do not want the AI to Enhance. Such as Blur, Smudges, or Pixels'), 
    gr.Slider(1, 15, 1, step=1, label='Guidance Scale: How much the AI influences the Upscaling.'), 
    gr.Slider(5, 50, 5, step=1, label='Number of Iterations'),
    gr.Slider(minimum=1, maximum=999999999999999999, randomize=True, step=1)], 
    outputs=gr.Image(type="filepath", label = 'Upscaled Image'), 
    title='SD Upscaler', 
    description='2x Latent Upscaler using SD 2.0 And 4x Upscaler using SD 2.1. This version runs on CPU or GPU and is currently running on a T4 GPU. For 4x Upscaling use images lower than 512x512, ideally 128x128 or smaller for 512x512 output. For 2x Upscaling use up to 512x512 images for 1024x1024 output.<br><br><b>Notice: Largest Accepted Resolution is 512x512', 
    article = "Code Monkey: <a href=\"https://huggingface.co/Manjushri\">Manjushri</a>").launch(max_threads=True, debug=True)