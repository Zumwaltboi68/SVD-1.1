import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import gradio as gr
import modin.pandas as pd 
from tempfile import NamedTemporaryFile

model = MusicGen.get_pretrained('large')

def genie(Prompt, Duration):
    model.set_generation_params(duration=Duration)
    wav = model.generate(Prompt)
    for idx, one_wav in enumerate(wav):
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name, one_wav.cpu(), model.sample_rate, strategy="loudness",
                loudness_headroom_db=16, add_suffix=False)                
    return file.name
    
title = 'MusicGen'
description = ("Audiocraft provides the code and models for MusicGen, a simple and controllable model for music generation. MusicGen is a single stage auto-regressive Transformer model trained over a 32kHz EnCodec tokenizer with 4 codebooks sampled at 50 Hz. Unlike existing methods like MusicLM, MusicGen doesn't not require a self-supervised semantic representation, and it generates all 4 codebooks in one pass. By introducing a small delay between the codebooks, we show we can predict them in parallel, thus having only 50 auto-regressive steps per second of audio.")
article = ('MusicGen consists of an EnCodec model for audio tokenization, an auto-regressive language model based on the transformer architecture for music modeling. The model comes in different sizes: 300M, 1.5B and 3.3B parameters ; and two variants: a model trained for text-to-music generation task and a model trained for melody-guided music generation. <br><br>Code Monkey: <a href=\"https://huggingface.co/Manjushri\">Manjushri</a>')
gr.Interface(fn=genie, inputs=[gr.Textbox(label='Text Prompt. Warning: Longer Prompts may cause reset.'), gr.Slider(minimum=1, maximum=8, value=6, label='Duration')], outputs=gr.Audio(), title=title, description=description, article=article).queue(max_size=2).launch(debug=True)    