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
gr.Interface(fn=upscale, inputs=[gr.Image(type='filepath', label='Low Resolution Image (less than 512x512, i.e. 128x128, 256x256, ect., ect..)'), gr.Textbox(label='Optional: Enter a Prompt to Slightly Guide the AI')], outputs=gr.Image(type='filepath'), title='SD 2.0 4x Upscaler', description='A 4x Low Resolution Upscaler using SD 2.0. Currently it takes about 15mins an images. <br>Expects a Lower than 512x512 image. <br><b>Warning: Images 512x512 or Higher Resolution WILL NOT BE UPSCALED and may result Quality Loss!', article = "Code Monkey: <a href=\"https://huggingface.co/Manjushri\">Manjushri</a>").launch(max_threads=True, debug=True)