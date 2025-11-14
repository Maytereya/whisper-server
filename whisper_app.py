import asyncio
import io
import os
import time
from typing import Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi import WebSocket
from faster_whisper import WhisperModel
from pydantic import BaseModel
from starlette.responses import JSONResponse

# -------- settings ----------
MODEL_NAME   = os.getenv("MODEL", "large-v3")        # tiny, base, small, medium, large-v3, distil-*
DEVICE       = os.getenv("DEVICE", "cuda")           # cpu | cuda
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8_float16")  # cuda: float16 / int8_float16; cpu: int8 / int8_float32
BEAM_SIZE_DEF = int(os.getenv("BEAM_SIZE", "5"))
VAD_MIN_SIL   = float(os.getenv("VAD_MIN_SILENCE", "0.5"))
LANG_DEFAULT  = os.getenv("LANGUAGE", "ru")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "8"))

app = FastAPI(title="Whisper Server (faster-whisper)")
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# ---- load prompt once ----
PROMPT_PATH = os.getenv("PROMPT_FILE", "/whisper_server/custom_prompt.txt")

_prompt_cache = {
    "text": "",
    "mtime": None,
}


def get_custom_prompt() -> str:
    global _prompt_cache

    try:
        st = os.stat(PROMPT_PATH)
    except FileNotFoundError:
        _prompt_cache["text"] = ""
        _prompt_cache["mtime"] = None
        return ""

    mtime = st.st_mtime
    if _prompt_cache["mtime"] != mtime:
        try:
            with open(PROMPT_PATH, "r", encoding="utf8") as f:
                text = f.read().strip()
        except Exception as e:
            print(f"⚠️ Failed to read prompt file: {e}")
            _prompt_cache["text"] = ""
            _prompt_cache["mtime"] = mtime
            return ""

        _prompt_cache["text"] = text
        _prompt_cache["mtime"] = mtime
        print(f"✅ Loaded custom prompt, len={len(text)}")

    return _prompt_cache["text"]


# ----------------------------

class PromptUpdate(BaseModel):
    prompt: str


@app.get("/prompt")
def get_prompt():
    """
    Вернуть текущий словарь (initial_prompt).
    """
    text = get_custom_prompt()
    return {
        "prompt": text,
        "length": len(text),
    }


@app.put("/prompt")
def update_prompt(body: PromptUpdate):
    """
    Обновить словарь: перезаписать файл и обновить кэш.
    """
    text = (body.prompt or "").strip()

    # пишем в файл
    try:
        with open(PROMPT_PATH, "w", encoding="utf8") as f:
            f.write(text + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка записи промпт - файла: {e}")

    # обновляем кэш
    try:
        st = os.stat(PROMPT_PATH)
        _prompt_cache["text"] = text
        _prompt_cache["mtime"] = st.st_mtime
    except Exception as e:
        print(f"⚠️ Failed to stat prompt file after write: {e}")
        _prompt_cache["text"] = text
        _prompt_cache["mtime"] = None

    return {
        "status": "ok",
        "length": len(text),
    }


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
    if sr != 16000:
        # либо ресемплить, либо бросать ошибку
        raise HTTPException(400, f"Ожидается 16kHz, пришло {sr}")
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

    prompt = get_custom_prompt()

    async with semaphore:
        audio, sr = await _read_audio_bytes_async(file)

        # параметры распознавания
        opts = dict(
            task=task,
            language=language or LANG_DEFAULT,
            beam_size=beam_size,
            vad_filter=vad,
            vad_parameters={"min_silence_duration_ms": int(VAD_MIN_SIL * 1000)},
            initial_prompt=prompt if prompt else None,
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
    prompt = get_custom_prompt()

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
                initial_prompt=prompt if prompt else None,
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
