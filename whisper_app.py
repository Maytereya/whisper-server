import io
import os
import time
from typing import Optional

import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from faster_whisper import WhisperModel
from starlette.responses import JSONResponse

# -------- settings ----------
MODEL_NAME = os.getenv("MODEL", "medium")  # tiny, base, small, medium, large-v3, distil-*
DEVICE = os.getenv("DEVICE", "cpu")  # cpu | cuda
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")  # cpu: int8/int8_float32, cuda: float16/int8_float16
BEAM_SIZE_DEF = int(os.getenv("BEAM_SIZE", "5"))
VAD_MIN_SIL = float(os.getenv("VAD_MIN_SILENCE", "0.5"))
LANG_DEFAULT = os.getenv("LANGUAGE", "ru")
# ----------------------------

app = FastAPI(title="Whisper Server (faster-whisper)")

t0 = time.time()
model = WhisperModel(
    MODEL_NAME,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    download_root=os.getenv("MODELS_DIR", "/models")
)
load_s = round(time.time() - t0, 2)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE, "compute_type": COMPUTE_TYPE, "loaded_sec": load_s}


def _read_audio_bytes(file: UploadFile) -> tuple:
    raw = file.file.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    # soundfile распознает формат, декодируем в float32, 16kHz mono
    data, sr = sf.read(io.BytesIO(raw), always_2d=False, dtype="float32")
    if sr != 16000:
        # ресемпл на коленке не делаем – пусть клиент присылает 16k, либо юзай ffmpeg на клиенте
        # если очень надо, подключай resampy/torchaudio и допиши ресемпл тут
        pass
    return data, sr


@app.post("/transcribe")
def transcribe(
        file: UploadFile = File(...),
        task: str = Query("transcribe", description="transcribe|translate"),
        language: Optional[str] = Query(None, description="например 'ru'"),
        beam_size: int = Query(BEAM_SIZE_DEF, ge=1, le=10),
        vad: bool = Query(True, description="включить встроенный VAD посекционно"),
) -> JSONResponse:
    audio, sr = _read_audio_bytes(file)

    # параметры распознавания
    opts = dict(
        task=task,
        language=language or LANG_DEFAULT,
        beam_size=beam_size,
        vad_filter=vad,
        vad_parameters={"min_silence_duration_ms": int(VAD_MIN_SIL * 1000)},
    )

    t1 = time.time()
    segments, info = model.transcribe(audio, **opts)
    text_parts = []
    segs = []
    for s in segments:
        text_parts.append(s.text)
        segs.append({
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "text": s.text
        })
    dur = round(time.time() - t1, 3)

    return JSONResponse({
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "language": info.language,
        "language_probability": info.language_probability,
        "time_sec": dur,
        "text": " ".join(text_parts).strip(),
        "segments": segs
    })


from fastapi import WebSocket
import numpy as np


@app.websocket("/stream")
async def websocket_stream(ws: WebSocket):
    """
    Простой WebSocket для потоковой передачи аудио чанков.
    Клиент отправляет бинарные чанки WAV/PCM, завершает 'END'.
    """
    await ws.accept()
    audio_buf = []
    try:
        while True:
            data = await ws.receive_bytes()
            if data == b"END":
                break
            samples, _ = sf.read(io.BytesIO(data), dtype="float32")
            audio_buf.append(samples)

        # склеиваем все полученные куски
        if not audio_buf:
            await ws.send_json({"error": "empty stream"})
            await ws.close()
            return

        audio = np.concatenate(audio_buf)
        t1 = time.time()
        segments, info = model.transcribe(
            audio,
            language=LANG_DEFAULT,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": int(VAD_MIN_SIL * 1000)},
        )
        text = " ".join([s.text for s in segments]).strip()
        dur = round(time.time() - t1, 3)

        await ws.send_json({
            "model": MODEL_NAME,
            "language": info.language,
            "text": text,
            "time_sec": dur,
        })
    except Exception as e:
        await ws.send_json({"error": str(e)})
    finally:
        await ws.close()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
