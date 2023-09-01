import gradio as gr
import torch
import numpy as np
import modin.pandas as pd
from PIL import Image
from diffusers import DiffusionPipeline 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    PYTORCH_CUDA_ALLOC_CONF={'max_split_size_mb': 8000}
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
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", use_safetensors=True)
    refiner = refiner.to(device)
    refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
      
def genie (prompt, negative_prompt, height, width, scale, steps, seed, upscaling, prompt_2, negative_prompt_2, high_noise_frac):
    #n_steps = 40
    generator = np.random.seed(0) if seed == 0 else torch.manual_seed(seed)
    int_image = pipe(prompt, prompt_2=prompt_2, negative_prompt=negative_prompt, negative_prompt_2=negative_prompt_2, num_inference_steps=steps, height=height, width=width, guidance_scale=scale, num_images_per_prompt=1, generator=generator, output_type="latent").images
    if upscaling == 'Yes':
        image = refiner(prompt=prompt, prompt_2=prompt_2, negative_prompt=negative_prompt, negative_prompt_2=negative_prompt_2, image=int_image, num_inference_steps=n_steps, denoising_start=high_noise_frac).images[0]
        upscaled = upscaler(prompt=prompt, negative_prompt=negative_prompt, image=image, num_inference_steps=5, guidance_scale=0).images[0]
        torch.cuda.empty_cache()
        return (image, upscaled)
    else:
        image = refiner(prompt=prompt, prompt_2=prompt_2, negative_prompt=negative_prompt, negative_prompt_2=negative_prompt_2, image=int_image, num_inference_steps=n_steps, denoising_start=high_noise_frac).images[0]   
        torch.cuda.empty_cache()
    return (image, image)

gr.Interface(fn=genie, inputs=[gr.Textbox(label='What you want the AI to generate. 77 Token Limit. A Token is Any Word, Number, Symbol, or Punctuation. Everything Over 77 Will Be Truncated!'), 
    gr.Textbox(label='What you Do Not want the AI to generate. 77 Token Limit'), 
    gr.Slider(512, 1024, 768, step=128, label='Height'),
    gr.Slider(512, 1024, 768, step=128, label='Width'),
    gr.Slider(1, 15, 10, step=.25, label='Guidance Scale: How Closely the AI follows the Prompt'), 
    gr.Slider(25, maximum=100, value=50, step=25, label='Number of Iterations'), 
    gr.Slider(minimum=0, step=1, maximum=999999999999999999, randomize=True, label='Seed: 0 is Random'),
    gr.Radio(['Yes', 'No'], value='No', label='Upscale?'),
    gr.Textbox(label='Embedded Prompt'),
    gr.Textbox(label='Embedded Negative Prompt'),
    gr.Slider(minimum=.7, maximum=.99, value=.95, step=.01, label='Refiner Denoise %')], 
    outputs=['image', 'image'],
    title="Stable Diffusion XL 1.0 GPU", 
    description="SDXL 1.0 GPU. <br><br><b>WARNING: Capable of producing NSFW (Softcore) images.</b>", 
    article = "If You Enjoyed this Demo and would like to Donate, you can send to any of these Wallets. <br>BTC: bc1qzdm9j73mj8ucwwtsjx4x4ylyfvr6kp7svzjn84 <br>3LWRoKYx6bCLnUrKEdnPo3FCSPQUSFDjFP <br>DOGE: DK6LRc4gfefdCTRk9xPD239N31jh9GjKez <br>SHIB (BEP20): 0xbE8f2f3B71DFEB84E5F7E3aae1909d60658aB891 <br>PayPal: https://www.paypal.me/ManjushriBodhisattva <br>ETH: 0xbE8f2f3B71DFEB84E5F7E3aae1909d60658aB891 <br>Code Monkey: <a href=\"https://huggingface.co/Manjushri\">Manjushri</a>").launch(debug=True, max_threads=80)
