import gradio as gr
import torch
import numpy as np
import modin.pandas as pd
from PIL import Image
from diffusers import DiffusionPipeline, StableDiffusionLatentUpscalePipeline

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.max_memory_allocated(device=device)
torch.cuda.empty_cache()

def genie (Model, Prompt, negative_prompt, height, width, scale, steps, seed, refine, high_noise_frac, upscale):
    generator = np.random.seed(0) if seed == 0 else torch.manual_seed(seed)
       
    if Model == "PhotoReal":
        pipe = DiffusionPipeline.from_pretrained("circulus/canvers-real-v3.8.1", torch_dtype=torch.float16, safety_checker=None) if torch.cuda.is_available() else DiffusionPipeline.from_pretrained("circulus/canvers-real-v3.8.1")
        pipe.enable_xformers_memory_efficient_attention()
        pipe = pipe.to(device)
        torch.cuda.empty_cache()
        if refine == "Yes":
            refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", use_safetensors=True, torch_dtype=torch.float16, variant="fp16") if torch.cuda.is_available() else DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0")
            refiner.enable_xformers_memory_efficient_attention()
            refiner = refiner.to(device)
            torch.cuda.empty_cache()
            int_image = pipe(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale).images
            image = refiner(Prompt, negative_prompt=negative_prompt, image=int_image, denoising_start=high_noise_frac).images[0]
            torch.cuda.empty_cache()
            if upscale == "Yes":
                refiner = DiffusionPipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, use_safetensors=True)
                refiner.enable_xformers_memory_efficient_attention()
                refiner = refiner.to(device)
                torch.cuda.empty_cache()
                upscaled = refiner(prompt=Prompt, negative_prompt=negative_prompt, image=image, num_inference_steps=15, guidance_scale=0).images[0]
                torch.cuda.empty_cache()
                return upscaled
            else:
                return image
        else:
            if upscale == "Yes":
                image = pipe(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale).images[0]
                upscaler = DiffusionPipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, use_safetensors=True)
                upscaler.enable_xformers_memory_efficient_attention()
                upscaler = upscaler.to(device)
                torch.cuda.empty_cache()
                upscaled = upscaler(prompt=Prompt, negative_prompt=negative_prompt, image=image, num_inference_steps=15, guidance_scale=0).images[0]
                torch.cuda.empty_cache()
                return upscaled
            else:
               image = pipe(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale).images[0]
               torch.cuda.empty_cache()
               return image
    
    if Model == "Anime":
        anime = DiffusionPipeline.from_pretrained("circulus/canvers-anime-v3.8.1", torch_dtype=torch.float16, safety_checker=None) if torch.cuda.is_available() else DiffusionPipeline.from_pretrained("circulus/canvers-anime-v3.8.1")
        anime.enable_xformers_memory_efficient_attention()
        anime = anime.to(device)
        torch.cuda.empty_cache()
        if refine == "Yes":
            refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", use_safetensors=True, torch_dtype=torch.float16, variant="fp16") if torch.cuda.is_available() else DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0")
            refiner.enable_xformers_memory_efficient_attention()
            refiner = refiner.to(device)
            torch.cuda.empty_cache()
            int_image = anime(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale).images
            image = refiner(Prompt, negative_prompt=negative_prompt, image=int_image, denoising_start=high_noise_frac).images[0]
            torch.cuda.empty_cache()
            if upscale == "Yes":
                refiner = DiffusionPipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, use_safetensors=True)
                refiner.enable_xformers_memory_efficient_attention()
                refiner = refiner.to(device)
                torch.cuda.empty_cache()
                upscaled = refiner(prompt=Prompt, negative_prompt=negative_prompt, image=image, num_inference_steps=15, guidance_scale=0).images[0]
                torch.cuda.empty_cache()
                return upscaled
            else:
                return image
        else:
            if upscale == "Yes":
                image = anime(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale).images[0]
                upscaler = DiffusionPipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, use_safetensors=True)
                upscaler.enable_xformers_memory_efficient_attention()
                upscaler = upscaler.to(device)
                torch.cuda.empty_cache()
                upscaled = upscaler(prompt=Prompt, negative_prompt=negative_prompt, image=image, num_inference_steps=15, guidance_scale=0).images[0]
                torch.cuda.empty_cache()
                return upscaled
            else:
               image = anime(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale).images[0]
               torch.cuda.empty_cache()
               return image
                
    if Model == "Disney":
        disney = DiffusionPipeline.from_pretrained("circulus/canvers-disney-v3.8.1", torch_dtype=torch.float16, safety_checker=None) if torch.cuda.is_available() else DiffusionPipeline.from_pretrained("circulus/canvers-disney-v3.8.1")
        disney.enable_xformers_memory_efficient_attention()
        disney = disney.to(device)
        torch.cuda.empty_cache()
        if refine == "Yes":
            refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", use_safetensors=True, torch_dtype=torch.float16, variant="fp16") if torch.cuda.is_available() else DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0")
            refiner.enable_xformers_memory_efficient_attention()
            refiner = refiner.to(device)
            torch.cuda.empty_cache()
            int_image = disney(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale).images
            image = refiner(Prompt, negative_prompt=negative_prompt, image=int_image, denoising_start=high_noise_frac).images[0]
            torch.cuda.empty_cache()
            
            if upscale == "Yes":
                refiner = DiffusionPipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, use_safetensors=True)
                refiner.enable_xformers_memory_efficient_attention()
                refiner = refiner.to(device)
                torch.cuda.empty_cache()
                upscaled = refiner(prompt=Prompt, negative_prompt=negative_prompt, image=image, num_inference_steps=15, guidance_scale=0).images[0]
                torch.cuda.empty_cache()
                return upscaled
            else:
                return image
        else:
            if upscale == "Yes":
                image = disney(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale).images[0]
                upscaler = DiffusionPipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, use_safetensors=True)
                upscaler.enable_xformers_memory_efficient_attention()
                upscaler = upscaler.to(device)
                torch.cuda.empty_cache()
                upscaled = upscaler(prompt=Prompt, negative_prompt=negative_prompt, image=image, num_inference_steps=15, guidance_scale=0).images[0]
                torch.cuda.empty_cache()
                return upscaled
            else: 
               image = disney(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale).images[0]
               torch.cuda.empty_cache()
               return image
            
    if Model == "StoryBook":
        story = DiffusionPipeline.from_pretrained("circulus/canvers-story-v3.8.1", torch_dtype=torch.float16, safety_checker=None) if torch.cuda.is_available() else DiffusionPipeline.from_pretrained("circulus/canvers-story-v3.8.1")
        story.enable_xformers_memory_efficient_attention()
        story = story.to(device)
        torch.cuda.empty_cache()
        if refine == "Yes":
            refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", use_safetensors=True, torch_dtype=torch.float16, variant="fp16") if torch.cuda.is_available() else DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0")
            refiner.enable_xformers_memory_efficient_attention()
            refiner = refiner.to(device)
            torch.cuda.empty_cache()
            int_image = story(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale).images
            image = refiner(Prompt, negative_prompt=negative_prompt, image=int_image, denoising_start=high_noise_frac).images[0]
            torch.cuda.empty_cache()
            
            if upscale == "Yes":
                refiner = DiffusionPipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, use_safetensors=True)
                refiner.enable_xformers_memory_efficient_attention()
                refiner = refiner.to(device)
                torch.cuda.empty_cache()
                upscaled = refiner(prompt=Prompt, negative_prompt=negative_prompt, image=image, num_inference_steps=15, guidance_scale=0).images[0]
                torch.cuda.empty_cache()
                return upscaled
            else:
                return image
        else:
            if upscale == "Yes":
                image = story(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale).images[0]
            
                upscaler = DiffusionPipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, use_safetensors=True)
                upscaler.enable_xformers_memory_efficient_attention()
                upscaler = upscaler.to(device)
                torch.cuda.empty_cache()
                upscaled = upscaler(prompt=Prompt, negative_prompt=negative_prompt, image=image, num_inference_steps=15, guidance_scale=0).images[0]
                torch.cuda.empty_cache()
                return upscaled
            else:
                
               image = story(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale).images[0]
               torch.cuda.empty_cache()
               return image

    if Model == "SemiReal":
        semi = DiffusionPipeline.from_pretrained("circulus/canvers-semi-v3.8.1", torch_dtype=torch.float16, safety_checker=None) if torch.cuda.is_available() else DiffusionPipeline.from_pretrained("circulus/canvers-semi-v3.8.1")
        semi.enable_xformers_memory_efficient_attention()
        semi = semi.to(device)
        torch.cuda.empty_cache()
        if refine == "Yes":
            refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", use_safetensors=True, torch_dtype=torch.float16, variant="fp16") if torch.cuda.is_available() else DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0")
            refiner.enable_xformers_memory_efficient_attention()
            refiner = refiner.to(device)
            torch.cuda.empty_cache()
            image = semi(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale).images
            image = refiner(Prompt, negative_prompt=negative_prompt, image=image, denoising_start=high_noise_frac).images[0]
            torch.cuda.empty_cache()
            
            if upscale == "Yes":
                refiner = DiffusionPipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, use_safetensors=True)
                refiner.enable_xformers_memory_efficient_attention()
                refiner = refiner.to(device)
                torch.cuda.empty_cache()
                upscaled = refiner(prompt=Prompt, negative_prompt=negative_prompt, image=image, num_inference_steps=15, guidance_scale=0).images[0]
                torch.cuda.empty_cache()
                return upscaled
            else:
                return image
        else:
            if upscale == "Yes":
                image = semi(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale).images[0]
            
                upscaler = DiffusionPipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, use_safetensors=True)
                upscaler.enable_xformers_memory_efficient_attention()
                upscaler = upscaler.to(device)
                torch.cuda.empty_cache()
                upscaled = upscaler(prompt=Prompt, negative_prompt=negative_prompt, image=image, num_inference_steps=15, guidance_scale=0).images[0]
                torch.cuda.empty_cache()
                return upscaled
            else:
                
                image = semi(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale).images[0]
                torch.cuda.empty_cache()
                return image

    if Model == "Animagine XL 3.0":
        animagine = DiffusionPipeline.from_pretrained("cagliostrolab/animagine-xl-3.0", torch_dtype=torch.float16, safety_checker=None) if torch.cuda.is_available() else DiffusionPipeline.from_pretrained("cagliostrolab/animagine-xl-3.0")
        animagine.enable_xformers_memory_efficient_attention()
        animagine = animagine.to(device)
        torch.cuda.empty_cache()
        if refine == "Yes":
            torch.cuda.empty_cache()
            torch.cuda.max_memory_allocated(device=device)
            int_image = animagine(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, output_type="latent").images
            torch.cuda.empty_cache()
            animagine = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", use_safetensors=True, torch_dtype=torch.float16, variant="fp16") if torch.cuda.is_available() else DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0")
            animagine.enable_xformers_memory_efficient_attention()
            animagine = animagine.to(device)
            torch.cuda.empty_cache()
            image = animagine(Prompt, negative_prompt=negative_prompt, image=int_image, denoising_start=high_noise_frac).images[0]
            torch.cuda.empty_cache()
            
            if upscale == "Yes":
                animagine = DiffusionPipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, use_safetensors=True)
                animagine.enable_xformers_memory_efficient_attention()
                animagine = animagine.to(device)
                torch.cuda.empty_cache()
                upscaled = animagine(prompt=Prompt, negative_prompt=negative_prompt, image=image, num_inference_steps=15, guidance_scale=0).images[0]
                torch.cuda.empty_cache()
                return upscaled
            else:
                return image
        else:
            if upscale == "Yes":
                image = animagine(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale).images[0]
            
                upscaler = DiffusionPipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, use_safetensors=True)
                upscaler.enable_xformers_memory_efficient_attention()
                upscaler = upscaler.to(device)
                torch.cuda.empty_cache()
                upscaled = upscaler(prompt=Prompt, negative_prompt=negative_prompt, image=image, num_inference_steps=15, guidance_scale=0).images[0]
                torch.cuda.empty_cache()
                return upscaled
            else:
                
               image = animagine(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale).images[0]
               torch.cuda.empty_cache()
               return image

    if Model == "SDXL 1.0":
        torch.cuda.empty_cache()
        torch.cuda.max_memory_allocated(device=device)
        sdxl = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        sdxl.enable_xformers_memory_efficient_attention()
        sdxl = sdxl.to(device)   
        torch.cuda.empty_cache()
    
        if refine == "Yes":
            torch.cuda.max_memory_allocated(device=device)
            torch.cuda.empty_cache()
            image = sdxl(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, output_type="latent").images
            torch.cuda.empty_cache()
            sdxl = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", use_safetensors=True, torch_dtype=torch.float16, variant="fp16") if torch.cuda.is_available() else DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0")
            sdxl.enable_xformers_memory_efficient_attention()
            sdxl = sdxl.to(device)
            torch.cuda.empty_cache()
            refined = sdxl(Prompt, negative_prompt=negative_prompt, image=image, denoising_start=high_noise_frac).images[0]
            torch.cuda.empty_cache()
            
            if upscale == "Yes":
                sdxl = DiffusionPipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, use_safetensors=True)
                sdxl.enable_xformers_memory_efficient_attention()
                sdxl = sdxl.to(device)
                torch.cuda.empty_cache()
                upscaled = sdxl(prompt=Prompt, negative_prompt=negative_prompt, image=refined, num_inference_steps=15, guidance_scale=0).images[0]
                torch.cuda.empty_cache()
                return upscaled
            else:
                return refined
        else:
            if upscale == "Yes":
                image = sdxl(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale).images[0]
            
                upscaler = DiffusionPipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, use_safetensors=True)
                upscaler.enable_xformers_memory_efficient_attention()
                upscaler = upscaler.to(device)
                torch.cuda.empty_cache()
                upscaled = upscaler(prompt=Prompt, negative_prompt=negative_prompt, image=image, num_inference_steps=15, guidance_scale=0).images[0]
                torch.cuda.empty_cache()
                return upscaled
            else:
                
               image = sdxl(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale).images[0]
               torch.cuda.empty_cache()
               
            
    return image
    
gr.Interface(fn=genie, inputs=[gr.Radio(['PhotoReal', 'Anime', 'Disney', 'StoryBook', 'SemiReal', 'Animagine XL 3.0', 'SDXL 1.0'], value='PhotoReal', label='Choose Model'),
                               gr.Textbox(label='What you want the AI to generate. 77 Token Limit.'), 
                               gr.Textbox(label='What you Do Not want the AI to generate. 77 Token Limit'),
                               gr.Slider(512, 1024, 768, step=128, label='Height'),
                               gr.Slider(512, 1024, 768, step=128, label='Width'),
                               gr.Slider(1, maximum=15, value=5, step=.25, label='Guidance Scale'), 
                               gr.Slider(25, maximum=100, value=50, step=25, label='Number of Iterations'), 
                               gr.Slider(minimum=0, step=1, maximum=9999999999999999, randomize=True, label='Seed: 0 is Random'), 
                               gr.Radio(["Yes", "No"], label='SDXL 1.0 Refiner: Use if the Image has too much Noise', value='No'),
                               gr.Slider(minimum=.9, maximum=.99, value=.95, step=.01, label='Refiner Denoise Start %'),
                               gr.Radio(["Yes", "No"], label = 'SD X2 Latent Upscaler?', value="No")],
             outputs=gr.Image(label='Generated Image'), 
             title="Manju Dream Booth V1.7 with SDXL 1.0 Refiner and SD X2 Latent Upscaler - GPU", 
             description="<br><br><b/>Warning: This Demo is capable of producing NSFW content.", 
             article = "If You Enjoyed this Demo and would like to Donate, you can send any amount to any of these Wallets. <br><br>BTC: bc1qzdm9j73mj8ucwwtsjx4x4ylyfvr6kp7svzjn84 <br>BTC2: 3LWRoKYx6bCLnUrKEdnPo3FCSPQUSFDjFP <br>DOGE: DK6LRc4gfefdCTRk9xPD239N31jh9GjKez <br>SHIB (BEP20): 0xbE8f2f3B71DFEB84E5F7E3aae1909d60658aB891 <br>PayPal: https://www.paypal.me/ManjushriBodhisattva <br>ETH: 0xbE8f2f3B71DFEB84E5F7E3aae1909d60658aB891 <br><br>Code Monkey: <a href=\"https://huggingface.co/Manjushri\">Manjushri</a>").launch(debug=True, max_threads=80)