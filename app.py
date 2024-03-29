import gradio as gr
import torch
import os
from glob import glob
from pathlib import Path
from typing import Optional

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image

import uuid
import random
from huggingface_hub import hf_hub_download

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.max_memory_allocated(device=device)
torch.cuda.empty_cache()
pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt-1-1", torch_dtype=torch.float16, use_safetensors=True, variant="fp16" )
pipe.to("cuda")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
torch.cuda.empty_cache()
max_64_bit_int = 2**63 - 1

def sample(
    image: Image,
    seed: Optional[int] = 42,
    randomize_seed: bool = True,
    motion_bucket_id: int = 127,
    fps_id: int = 6,
    version: str = "svd_xt_1-1",
    cond_aug: float = 0.02,
    decoding_t: int = 3,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: str = "outputs",
):
    if image.mode == "RGBA":
        image = image.convert("RGB")
        
    if(randomize_seed):
        seed = random.randint(0, max_64_bit_int)
    generator = torch.manual_seed(seed)
    
    os.makedirs(output_folder, exist_ok=True)
    base_count = len(glob(os.path.join(output_folder, "*.mp4")))
    video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")

    frames = pipe(image, decode_chunk_size=decoding_t, generator=generator, motion_bucket_id=motion_bucket_id, noise_aug_strength=1, num_frames=25).frames[0]
    export_to_video(frames, video_path, fps=fps_id)
    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    return video_path, seed

def resize_image(image, output_size=(1024, 576)):
    # Calculate aspect ratios
    target_aspect = output_size[0] / output_size[1]  # Aspect ratio of the desired size
    image_aspect = image.width / image.height  # Aspect ratio of the original image

    # Resize then crop if the original image is larger
    if image_aspect > target_aspect:
        # Resize the image to match the target height, maintaining aspect ratio
        new_height = output_size[1]
        new_width = int(new_height * image_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        # Calculate coordinates for cropping
        left = (new_width - output_size[0]) / 2
        top = 0
        right = (new_width + output_size[0]) / 2
        bottom = output_size[1]
    else:
        # Resize the image to match the target width, maintaining aspect ratio
        new_width = output_size[0]
        new_height = int(new_width / image_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        # Calculate coordinates for cropping
        left = 0
        top = (new_height - output_size[1]) / 2
        right = output_size[0]
        bottom = (new_height + output_size[1]) / 2

    torch.cuda.empty_cache()
    cropped_image = resized_image.crop((left, top, right, bottom))
    return cropped_image

with gr.Blocks() as demo:
  gr.Markdown("Stable Video Diffusion (SVD) 1.1 Image-to-Video is a diffusion model that takes in a still image as a conditioning frame, and generates a video from it. https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1 <br> This model was trained to generate 25 frames at resolution 1024x576 given a context frame of the same size, finetuned from SVD Image-to-Video [25 frames].")
  with gr.Row():
    with gr.Column():
        image = gr.Image(label="Upload your image", type="pil")
        generate_btn = gr.Button("Generate")
    video = gr.Video()
  with gr.Accordion("Advanced options", open=False):
      seed = gr.Slider(label="Seed", value=42, randomize=True, minimum=0, maximum=max_64_bit_int, step=1)
      randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
      motion_bucket_id = gr.Slider(label="Motion bucket id", info="Controls how much motion to add/remove from the image", value=60, minimum=1, maximum=255)
      fps_id = gr.Slider(label="Frames per second", info="The length of your video in seconds will be 25/fps", value=5, minimum=5, maximum=30)
  gr.Markdown("If You Enjoyed this Demo and would like to Donate, you can send any amount to any of these Wallets. <br><br>BTC: bc1qzdm9j73mj8ucwwtsjx4x4ylyfvr6kp7svzjn84 <br>BTC2: 3LWRoKYx6bCLnUrKEdnPo3FCSPQUSFDjFP <br>DOGE: DK6LRc4gfefdCTRk9xPD239N31jh9GjKez <br>SHIB (BEP20): 0xbE8f2f3B71DFEB84E5F7E3aae1909d60658aB891 <br>PayPal: https://www.paypal.me/ManjushriBodhisattva <br>ETH: 0xbE8f2f3B71DFEB84E5F7E3aae1909d60658aB891 <br><br>Code Monkey: <a href=\"https://huggingface.co/Manjushri\">Manjushri</a>")      
  image.upload(fn=resize_image, inputs=image, outputs=image, queue=False)
  generate_btn.click(fn=sample, inputs=[image, seed, randomize_seed, motion_bucket_id, fps_id], outputs=[video, seed], api_name="video")
  

if __name__ == "__main__":
    demo.queue(max_size=20, api_open=False)
    demo.launch(show_api=False)