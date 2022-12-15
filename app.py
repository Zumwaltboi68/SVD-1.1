import torch #needed only for GPU
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import gradio as gr
# load model for CPU or GPU
model_id = "stabilityai/stable-diffusion-x4-upscaler"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16") if torch.cuda.is_available() else StableDiffusionUpscalePipeline.from_pretrained(model_id)
pipe = pipe.to(device)
#define interface 
def upscale(low_res_img, prompt, negative_prompt):
 low_res_img = Image.open(low_res_img).convert("RGB")
 low_res_img = low_res_img.resize((128, 128))
 upscaled_image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=low_res_img, guidance_scale=2, num_inference_steps=20).images[0]
 #upscaled_image.save("upsampled.png")
 return upscaled_image
#launch interface
gr.Interface(fn=upscale, inputs=[gr.Image(type='filepath', label='Low Resolution Image (less than 512x512, i.e. 128x128, 256x256, ect., ect..)'), gr.Textbox(label='Optional: Enter a Prompt to Slightly Guide the AI'), gr.Textbox(label='Slightly influence What you do not want to see.')], outputs=gr.Image(type='filepath'), title='SD 2.0 4x Upscaler', description='A 4x Low Resolution Upscaler using SD 2.0. Currently it takes about 15mins an image. <br>Expects a Lower than 512x512 image. <br><br><b>Warning: Images 512x512 or Higher Resolution WILL NOT BE UPSCALED and may result Quality Loss!', article = "Code Monkey: <a href=\"https://huggingface.co/Manjushri\">Manjushri</a>").launch(max_threads=True, debug=True)