import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
# TODO: Try using transformers pipeline instead of dierctly loading model; see if it improves performance because 2 mins for voice is way too long...
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-large-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-large-v1")

prompt = "Sigh... Again with the \"hello\"? You know, I was having such a nice conversation about ice before, and now you come along and ruin it with yet another greeting. Can't you just surprise me for once? But fine, if it'll make you happy, I'll play along. \"Hello Alyssa\"... Wow, so original. You know what would've been nice instead of saying that? Maybe a simple \"I love you\" or something, like from the heart."
description = "A female voice that is gentle and soft, speaking at a slow pace, with a moderate pitch. The recording is of very high quality."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
