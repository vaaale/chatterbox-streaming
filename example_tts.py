import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# model = ChatterboxTTS.from_pretrained(device=device)
model = ChatterboxTTS.from_local(ckpt_dir="/mnt/storage/Models/Chatterbox/checkpoints/chatterbox_finetuned_norwegian", device=device)

text = "Dette er en test av Chatterbox TTS-modellen. Vi skal se hvordan den håndterer norsk tekst."

exaggeration = 0.5
temperature = 0.5
cfgw = 0.3  # CFG weight, adjust as needed

wav = model.generate(
    text,
    exaggeration = exaggeration,
    temperature = temperature,
    cfg_weight = cfgw,
)
ta.save("no-test-1.wav", wav, model.sr)

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "voices/henrik.wav"

wav = model.generate(
    text,
    exaggeration=exaggeration,
    temperature=temperature,
    cfg_weight=cfgw,
    audio_prompt_path=AUDIO_PROMPT_PATH
)
ta.save("no-test-2.wav", wav, model.sr)
