import torchaudio as ta
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Load the Multilingual model
model = ChatterboxMultilingualTTS.from_pretrained(device='cpu')

# Generate with Paralinguistic Tags
text = "The quick brown fox jumps over the lazy dog. In other news, I think war never changes, and neither does mankind."

# Generate audio (requires a reference clip for voice cloning)
wav = model.generate(text, language_id="en")
# wav = model.generate(text, audio_prompt_path="voices/Kuklina.wav", language_id="it")
ta.save("test-multi-2.wav", wav, model.sr)
