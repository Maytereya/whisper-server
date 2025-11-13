import asyncio
import io
import os
import time
from typing import Optional

import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from faster_whisper import WhisperModel
from starlette.responses import JSONResponse
from fastapi import WebSocket
import numpy as np

# -------- settings ----------
MODEL_NAME = os.getenv("MODEL", "medium")  # tiny, base, small, medium, large-v3, distil-*
DEVICE = os.getenv("DEVICE", "cpu")  # cpu | cuda
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")  # cpu: int8/int8_float32, cuda: float16/int8_float16
BEAM_SIZE_DEF = int(os.getenv("BEAM_SIZE", "5"))
VAD_MIN_SIL = float(os.getenv("VAD_MIN_SILENCE", "0.5"))
LANG_DEFAULT = os.getenv("LANGUAGE", "ru")
MAX_CONCURRENT = 8

# ---- load prompt once ----
PROMPT_PATH = os.getenv("PROMPT_FILE", "custom_prompt.txt")
try:
    with open(PROMPT_PATH, "r", encoding="utf8") as f:
        CUSTOM_PROMPT = f.read().strip()
except:
    CUSTOM_PROMPT = ""
    print("⚠️ No custom prompt found, continuing without initial_prompt")

# ----------------------------

app = FastAPI(title="Whisper Server (faster-whisper)")
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

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
    return {"name": "neiry.ai agent", "status": "ok", "model": MODEL_NAME, "device": DEVICE,
            "compute_type": COMPUTE_TYPE, "loaded_sec": load_s}


async def _read_audio_bytes_async(file: UploadFile) -> tuple:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "ничего не загрузилось в Whisper")
    data, sr = sf.read(io.BytesIO(raw), always_2d=False, dtype="float32")
    return data, sr


@app.post("/transcribe")
async def transcribe(
        file: UploadFile = File(...),
        task: str = Query("transcribe", description="transcribe|translate"),
        language: Optional[str] = Query(None, description="например 'ru'"),
        beam_size: int = Query(BEAM_SIZE_DEF, ge=1, le=10),
        vad: bool = Query(True, description="включить встроенный VAD посекционно"),
) -> JSONResponse:

    if file is None:
        raise HTTPException(400, "no file")
    async with semaphore:
        audio, sr = await _read_audio_bytes_async(file)

        # параметры распознавания
        opts = dict(
            task=task,
            language=language or LANG_DEFAULT,
            beam_size=beam_size,
            vad_filter=vad,
            vad_parameters={"min_silence_duration_ms": int(VAD_MIN_SIL * 1000)},
            initial_prompt=CUSTOM_PROMPT if CUSTOM_PROMPT else None,
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



@app.websocket("/stream")
async def websocket_stream(ws: WebSocket):
    """
    Стриминговый WebSocket:
    - клиент шлёт бинарные чанки audio (float32, 16 kHz, mono)
    - по завершении шлёт текстовое сообщение 'END' ИЛИ просто закрывает соединение
    - сервер склеивает аудио и один раз прогоняет через Whisper
    """
    await ws.accept()
    audio_buf: list[np.ndarray] = []

    try:
        while True:
            msg = await ws.receive()

            # Клиент закрыл соединение
            if msg["type"] == "websocket.disconnect":
                break

            # Текстовое сообщение (например, 'END')
            text = msg.get("text")
            if text is not None:
                if text.strip().upper() == "END":
                    break
                # можно игнорировать прочий текст или логировать
                continue

            # Бинарные данные — аудиочанк
            data = msg.get("bytes")
            if data:
                # ожидаем float32 PCM, mono, 16 kHz
                # если отправляется int16 — нужно заменить dtype
                chunk = np.frombuffer(data, dtype=np.float32)
                if chunk.size > 0:
                    audio_buf.append(chunk)

        if not audio_buf:
            await ws.send_json({"error": "empty stream"})
            return

        audio = np.concatenate(audio_buf)

        t1 = time.time()
        # ограничиваем одновременно работающие распознавания
        async with semaphore:
            segments, info = model.transcribe(
                audio,
                task="transcribe",
                language=LANG_DEFAULT,
                beam_size=BEAM_SIZE_DEF,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": int(VAD_MIN_SIL * 1000)},
                initial_prompt=CUSTOM_PROMPT if CUSTOM_PROMPT else None,
            )

        text_parts = []
        segs = []
        for s in segments:
            text_parts.append(s.text)
            segs.append(
                {
                    "start": round(s.start, 3),
                    "end": round(s.end, 3),
                    "text": s.text,
                }
            )
        dur = round(time.time() - t1, 3)

        await ws.send_json(
            {
                "model": MODEL_NAME,
                "device": DEVICE,
                "compute_type": COMPUTE_TYPE,
                "language": info.language,
                "language_probability": info.language_probability,
                "time_sec": dur,
                "text": " ".join(text_parts).strip(),
                "segments": segs,
            }
        )

    except Exception as e:
        # на всякий случай не падаем молча
        await ws.send_json({"Whisper websocket error": str(e)})
    finally:
        await ws.close()


if __name__ == "__main__":
    uvicorn.run("whisper_app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
