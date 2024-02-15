import gradio as gr
import torch
import os
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
from huggingface_hub import hf_hub_download

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.max_memory_allocated(device=device)
torch.cuda.empty_cache()
pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt-1-1", torch_dtype=torch.float16, use_safetensors=True, variant="fp16" )
pipe.to("cuda")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
torch.cuda.empty_cache()

def genie(image):
    torch.cuda.empty_cache()
    if image.mode == "RGBA":
        image = image.convert("RGB")
    frames = pipe(image=image, decode_chunk_size=3).frames[0]
    export_to_video(frames, fps=6)
    torch.cuda.empty_cache()
    return frames

gr.Interface(fn=genie, inputs=gr.Image(type="numpy"), outputs=gr.Video()).launch(debug=True, max_threads=80)