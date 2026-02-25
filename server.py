import tempfile
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from chatterbox.tts_turbo import ChatterboxTurboTTS

app = FastAPI(title="Chatterbox Turbo TTS API")


def select_device(min_vram_gb: float = 6.0) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
    except Exception:
        return "cpu"
    return "cuda" if total_vram_bytes > (min_vram_gb * (1024**3)) else "cpu"


model = ChatterboxTurboTTS.from_pretrained(device=select_device())
BASE_DIR = Path(__file__).resolve().parent
VOICES_DIR = BASE_DIR / "voices"


def tensor_to_pcm16le_bytes(audio: torch.Tensor) -> bytes:
    pcm = audio.squeeze().detach().cpu().numpy().astype(np.float32)
    pcm = np.clip(pcm, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    return pcm_i16.tobytes()


@app.post("/tts")
async def tts(
    text: str = Form(...),
    reference_wav: UploadFile | None = File(default=None),
) -> Response:
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="`text` must be non-empty.")

    generate_kwargs = {}
    temp_wav_path: str | None = None
    if reference_wav is not None:
        uploaded = await reference_wav.read()
        if not uploaded:
            raise HTTPException(status_code=400, detail="`reference_wav` was provided but is empty.")
        if reference_wav.filename and not reference_wav.filename.lower().endswith(".wav"):
            raise HTTPException(status_code=400, detail="`reference_wav` must be a .wav file.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded)
            temp_wav_path = tmp.name
        generate_kwargs["audio_prompt_path"] = temp_wav_path

    try:
        wav = model.generate(text, **generate_kwargs)
    finally:
        if temp_wav_path is not None:
            Path(temp_wav_path).unlink(missing_ok=True)

    pcm_bytes = tensor_to_pcm16le_bytes(wav)
    return Response(
        content=pcm_bytes,
        media_type="audio/pcm",
        headers={
            "X-Audio-Format": "pcm_s16le",
            "X-Sample-Rate": str(model.sr),
            "X-Channels": "1",
        },
    )


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
