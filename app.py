import gradio as gr
import torch
import os
from glob import glob
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

output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)
base_count = len(glob(os.path.join(output_folder, "*.mp4")))
video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
return video_path    

def genie(image):
    torch.cuda.empty_cache()
    if image.mode == "RGBA":
        image = image.convert("RGB")
    frames = pipe(image=image, decode_chunk_size=3).frames[0]
    export_to_video(frames, video_path, fps=6)
    torch.cuda.empty_cache()
    return frames

gr.Interface(fn=genie, inputs=gr.Image(type='pil'), outputs=gr.Video()).launch(debug=True, max_threads=80)