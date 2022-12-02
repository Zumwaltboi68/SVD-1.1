import torch #needed only for GPU
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import gradio as gr
# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id)
pipeline = pipeline.to("cpu")
#define interface 
def upscale(low_res_img, prompt):
 low_res_img = Image.open(low_res_img).convert("RGB")
 low_res_img = low_res_img.resize((128, 128))
 upscaled_image = pipeline(prompt=prompt, image=low_res_img, guidance_scale=1, num_inference_steps=50).images[0]
 upscaled_image.save("upsampled.png")
 return upscaled_image
#launch interface
gr.Interface(fn=upscale, inputs=[gr.Image(type='filepath'), 'text'], outputs=gr.Image(type='filepath')).launch(max_threads=True, debug=True)