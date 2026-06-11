# Whisper GPU Server

Этот проект поднимает локальный HTTP/WebSocket API для распознавания речи через
`faster-whisper` в GPU-контейнере. Контейнер используется как ASR-сервис для
локального AI-контура, в частности для `LocalRAGagent0.1`.

Секретные значения в документации не фиксируются. API-ключ хранится локально в
`config/secrets.ini` и монтируется в контейнер как read-only файл.

## Состав проекта

- `docker-compose.yml` - описание сервиса `whisper-gpu`, GPU, портов, volume и сети.
- `Dockerfile` - CUDA/Python образ, установка зависимостей и запуск `uvicorn`.
- `whisper_app.py` - FastAPI-приложение с REST и WebSocket endpoint.
- `load_secrets.py` - чтение API-ключа из `secrets.ini`.
- `custom_prompt.txt` - словарь терминов/фамилий для `initial_prompt` Whisper.
- `requirements.txt` - Python-зависимости.
- `config/secrets.ini` - локальный файл секретов, не должен попадать в git.

## Docker-сервис

Compose поднимает один сервис:

```yaml
services:
  whisper-gpu:
    container_name: whisper-gpu
    ports:
      - "8001:8000"
    networks:
      - local_net
```

Внешний адрес с хоста: `http://<server>:8001`.

Адрес внутри Docker-сети `local_net`: `http://whisper-gpu:8000`.

Сеть `local_net` объявлена как внешняя, поэтому она должна существовать заранее:

```bash
docker network create local_net
```

Контейнер требует NVIDIA Docker/runtime. В compose указан доступ ко всем GPU через
`deploy.resources.reservations.devices`, но дополнительно ограничена видимость:

```yaml
CUDA_VISIBLE_DEVICES: "2"
```

Это означает, что сервис должен работать на физической GPU с индексом `2`
(третья GPU на хосте). Внутри контейнера эта GPU может отображаться как
единственная доступная CUDA-карта.

## Модель и параметры распознавания

Параметры задаются переменными окружения в `docker-compose.yml`:

```yaml
MODEL: "large-v3"
DEVICE: "cuda"
COMPUTE_TYPE: "float16"
LANGUAGE: "ru"
PROMPT_FILE: "/whisper_server/custom_prompt.txt"
```

Дополнительные параметры, поддержанные кодом:

- `BEAM_SIZE`, default `5`
- `VAD_MIN_SILENCE`, default `0.5`
- `MAX_CONCURRENT`, default `8`
- `MODELS_DIR`, default `/models`

Модель загружается при старте приложения:

```python
WhisperModel(
    MODEL_NAME,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    download_root=os.getenv("MODELS_DIR", "/models"),
)
```

Кэш моделей хранится в named volume `whisper_models:/models`. Первый запуск может
скачивать модель из Hugging Face и занять заметное время.

## Volumes

```yaml
volumes:
  - whisper_models:/models
  - ./custom_prompt.txt:/whisper_server/custom_prompt.txt
  - ./config/secrets.ini:/whisper_server/secrets.ini:ro
```

Назначение:

- `whisper_models` - постоянный кэш модели.
- `custom_prompt.txt` - словарь/подсказка для повышения качества распознавания.
- `config/secrets.ini` - API-ключ для защищенных endpoint.

Формат `config/secrets.ini`:

```ini
[whisper]
api_key = <WHISPER_API_KEY>
```

Файл `load_secrets.py` по умолчанию ищет `secrets.ini` в рабочей директории.
В контейнере `WORKDIR=/whisper_server`, поэтому путь совпадает с mount path
`/whisper_server/secrets.ini`.

При прямом запуске приложения на хосте без Docker нужно либо положить
`secrets.ini` в корень проекта, либо доработать путь загрузки секрета.

## Запуск

На сервере с Docker и NVIDIA runtime:

```bash
docker compose up -d --build
```

Проверка статуса:

```bash
curl -fsS http://localhost:8001/healthz
```

Просмотр логов:

```bash
docker logs -f whisper-gpu
```

Остановка:

```bash
docker compose down
```

## API

Важно: `uvicorn` запускается с `--root-path /whisper`, и `FastAPI` тоже создан с
`root_path="/whisper"`. Это не добавляет префикс к маршрутам при прямом обращении.
Без reverse proxy фактические пути остаются `/healthz`, `/transcribe`, `/stream`,
`/prompt`.

Если сервис публикуется за reverse proxy по префиксу `/whisper`, proxy должен
корректно обрабатывать этот префикс. Текущий compose healthcheck проверяет прямой
путь `http://localhost:8000/healthz` внутри контейнера.

### `GET /healthz`

Проверка готовности сервиса. API-ключ не требуется.

Пример:

```bash
curl -s http://localhost:8001/healthz | jq .
```

Ответ содержит:

- `name`
- `status`
- `model`
- `device`
- `compute_type`
- `loaded_sec`

### `POST /transcribe`

REST-распознавание загруженного аудиофайла.

Требует заголовок:

```http
X-API-Key: <WHISPER_API_KEY>
```

Query-параметры:

- `task` - `transcribe` или `translate`, default `transcribe`
- `language` - язык, например `ru`; если не задан, используется `LANGUAGE`
- `beam_size` - от `1` до `10`, default из `BEAM_SIZE`
- `vad` - `true`/`false`, default `true`

Ожидается аудио с sample rate `16000 Hz`. Если пришел другой sample rate, сервер
вернет `400`.

Пример:

```bash
curl -s "http://localhost:8001/transcribe?task=transcribe" \
  -H "X-API-Key: <WHISPER_API_KEY>" \
  -F "file=@sample.wav" \
  | jq .
```

Ответ содержит:

- `model`
- `device`
- `compute_type`
- `language`
- `language_probability`
- `time_sec`
- `text`
- `segments[]` с `start`, `end`, `text`

### `WebSocket /stream`

WebSocket endpoint для отправки аудиопотока. В текущей реализации клиент обычно
отправляет весь фрагмент одним бинарным сообщением, затем текстовое сообщение
`END`.

Авторизация передается query-параметром:

```text
ws://<server>:8001/stream?api_key=<WHISPER_API_KEY>
```

Формат бинарных сообщений:

- PCM `float32`
- mono
- `16000 Hz`

Сервер склеивает все бинарные чанки, один раз запускает `model.transcribe(...)` и
возвращает JSON того же вида, что `/transcribe`.

### `GET /prompt`

Возвращает текущий `initial_prompt` из `custom_prompt.txt`.

На момент анализа endpoint не проверяет API-ключ, хотя клиент из
`LocalRAGagent0.1` отправляет `X-API-Key` при чтении.

Пример:

```bash
curl -s http://localhost:8001/prompt | jq .
```

### `PUT /prompt`

Перезаписывает `custom_prompt.txt` и обновляет in-memory cache.

На момент анализа endpoint не проверяет API-ключ. Это значит, что любой клиент,
имеющий сетевой доступ к сервису, может изменить словарь распознавания.

Пример:

```bash
curl -s -X PUT http://localhost:8001/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Фамилии: ...\nПроцедуры: ..."}' \
  | jq .
```

## Интеграция с `LocalRAGagent0.1`

В соседнем проекте `/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1` найдены
клиенты этого сервиса:

- `whisper/wisper_ws_client.py`
- `whisper/whisper_dict.py`
- `agent_logic_2/config.py`
- `agent_logic_2/config.ini`
- `agent_logic_2/gradio_ui/tabs/system_settings_tab.py`
- `agent_logic_2/gradio_ui/tabs/assistant_tab.py`

Конфигурация адресов в `agent_logic_2/config.ini`:

```ini
[APP]
environment = PRODUCTION

[ASR_WHISPER.PRODUCTION]
ws_url = ws://172.16.0.16:8001/stream
http_api = http://172.16.0.16:8001

[ASR_WHISPER.DOCKER_PRODUCTION]
ws_url = ws://whisper-gpu:8000/stream
http_api = http://whisper-gpu:8000
```

Смысл адресов:

- `PRODUCTION` - обращение к Whisper-сервису по IP хоста AI-сервера и
  опубликованному порту `8001`.
- `DOCKER_PRODUCTION` - обращение из контейнеров в той же Docker-сети
  `local_net` по имени сервиса `whisper-gpu` и внутреннему порту `8000`.

`WHISPER_API_KEY` в `LocalRAGagent0.1` должен совпадать со значением
`[whisper] api_key` из этого проекта.

WebSocket-клиент `wisper_ws_client.py` делает следующее:

1. Получает аудио из Gradio в формате `(sample_rate, numpy.ndarray)`.
2. Приводит stereo к mono.
3. Нормализует integer PCM в `float32`.
4. Ресемплит аудио до `16000 Hz`.
5. Подключается к `WHISPER_URL?api_key=<WHISPER_API_KEY>`.
6. Отправляет `data.tobytes()`.
7. Отправляет `END`.
8. Получает JSON и возвращает поле `text`.

Модуль `whisper_dict.py` работает с `/prompt`:

- `load_prompt()` читает словарь через `GET <WHISPER_HTTP_API>/prompt`.
- `save_prompt(text)` сохраняет словарь через `PUT <WHISPER_HTTP_API>/prompt`.

В UI `LocalRAGagent0.1` словарь Whisper доступен в системных настройках, а
голосовой ввод в assistant tab автоматически вызывает WebSocket-распознавание
после завершения записи.

## Проверка живого сервера

Проверка выполнена `2026-06-09` по SSH через alias `ai-server-singbox` из
`~/.ssh/config`.

Важно: адрес `172.16.4.68`, указанный как OpenVPN-адрес, из текущей локальной
среды не отвечал. Локальный gateway `172.16.4.1` возвращал
`Destination Net Unreachable`. С самого AI-сервера маршрут до `172.16.4.68`
тоже давал `Destination Host Unreachable`. Рабочий SSH-доступ был через
`172.16.0.16`:

```text
host: interrupt
user: vl
primary ip: 172.16.0.16/22
```

Фактический путь проекта на сервере:

```text
/home/vl/bookworm_app/whisper-server
```

Фактический контейнер:

```text
name: whisper-gpu
compose project: whisper-server
image: whisper-server-whisper-gpu
status: running / healthy
started: 2026-04-27T08:32:04Z
restart count: 0
published port: 0.0.0.0:8001 -> 8000/tcp
docker network: local_net
container ip: 172.19.0.12
```

Проверенный `healthz`:

```json
{
  "name": "neiry.ai agent",
  "status": "ok",
  "model": "large-v3",
  "device": "cuda",
  "compute_type": "float16",
  "loaded_sec": 6.31
}
```

Проверенный `/transcribe` smoke test с 1 секундой тишины:

```text
http_status=200
model=large-v3
device=cuda
compute_type=float16
language=ru
time_sec=0.047
text=""
segments=0
```

Секрет:

```text
/home/vl/bookworm_app/whisper-server/config/secrets.ini
[whisper]
api_key=<redacted> (39 chars)
```

Фактический `custom_prompt.txt` на сервере отличается от git HEAD и от локальной
копии: в нем расширен список фамилий и процедур. Через API длина текущего prompt:

```text
GET /prompt -> length=349
```

Удаленный git status на сервере грязный:

```text
tracked files: mode changes 100644 -> 100755
custom_prompt.txt: content changed
config/secrets.ini: untracked
```

Хэши основных файлов на сервере совпали с локальными для:

- `Dockerfile`
- `docker-compose.yml`
- `load_secrets.py`
- `whisper_app.py`
- `requirements.txt`

Не совпал только `custom_prompt.txt`.

### Живые URL

Прямой доступ к контейнеру/host port:

```text
http://localhost:8001/healthz          -> 200
http://localhost:8001/docs             -> 200
http://localhost:8001/openapi.json     -> 200
http://localhost:8001/whisper/healthz  -> 404
http://localhost:8001/whisper/docs     -> 404
```

OpenAPI объявляет server URL `/whisper`, но реальные прямые пути в приложении:

```text
/healthz
/prompt
/transcribe
```

`/stream` есть как WebSocket endpoint, поэтому не отображается в OpenAPI paths.

### Nginx

В `Distributed-Local-AI-Agent/nginx.conf` на сервере есть upstream и location для
Whisper:

```nginx
upstream whisper_api {
    server whisper-gpu:8000;
}

location ^~ /whisper/ {
    allow 172.16.4.0/24;
    allow 192.168.0.35;
    allow 192.168.0.0/22;
    deny all;

    proxy_pass http://whisper_api/;
}
```

То есть внешний nginx-путь должен быть `/whisper/...`, а nginx срезает префикс и
отправляет запрос внутрь контейнера как `/...`. При локальном запросе с самого
сервера `http://localhost/whisper/healthz` nginx вернул `403`, потому что
`127.0.0.1` не входит в allowlist. Для клиентов из `172.16.4.0/24` или
`192.168.0.0/22` этот путь должен быть разрешен.

### Живые клиенты

В запущенных контейнерах `bookworm-agent` и `agent-api` активна среда:

```text
environment=DOCKER_PRODUCTION
WHISPER_URL=ws://whisper-gpu:8000/stream
WHISPER_HTTP_API=http://whisper-gpu:8000
WHISPER_API_KEY_LEN=39
```

Проверка из `bookworm-agent`:

```text
GET http://whisper-gpu:8000/healthz -> 200
GET http://whisper-gpu:8000/prompt  -> 200
GET http://whisper-gpu:8000/whisper/healthz -> 404
```

### GPU на сервере

На сервере три `NVIDIA GeForce RTX 4090`. `whisper-gpu` запущен с:

```text
CUDA_VISIBLE_DEVICES=2
NVIDIA_VISIBLE_DEVICES=all
```

С хоста видно, что процесс `python -m uvicorn ... whisper_app:app` держит память
на GPU 2:

```text
GPU 2: python PID 2592, used_memory ~3488 MiB
```

Остальную VRAM на всех трех GPU занимает `ollama`. На момент проверки:

```text
GPU 0: used ~19615 MiB, free ~4493 MiB
GPU 1: used ~19565 MiB, free ~4543 MiB
GPU 2: used ~19444 MiB, free ~4664 MiB
```

Отдельный `docker exec whisper-gpu nvidia-smi` возвращал
`Failed to initialize NVML: Unknown Error`, а новый `docker exec` Python-процесс
не видел CUDA. При этом уже запущенный серверный процесс держит GPU-память и
успешно обслуживает `/transcribe`. Для диагностики GPU в этом контейнере лучше
смотреть процессы через `nvidia-smi` на хосте, а не через `docker exec`.

## Схема сетевого взаимодействия

```text
Host/client
  |
  | http://172.16.0.16:8001
  v
Docker published port 8001
  |
  | container port 8000
  v
whisper-gpu
  |
  | FastAPI + faster-whisper
  v
GPU index 2, model cache /models
```

Для контейнеров в `local_net`:

```text
bookworm-agent / agent-api / other container
  |
  | http://whisper-gpu:8000
  v
whisper-gpu
```

## Наблюдения и риски

- Локальный `docker` на рабочей машине недоступен, но живой контейнер проверен
  по SSH на сервере `interrupt` / `172.16.0.16`.
- Заявленный OpenVPN-адрес `172.16.4.68` на момент проверки был недоступен:
  gateway возвращал `Destination Net/Host Unreachable`.
- `config/secrets.ini` есть локально и не отслеживается git. Это правильно; не
  коммитить реальные ключи.
- В проекте нет `.dockerignore`. Даже если секрет не копируется в image, Docker
  build context может отправлять `config/secrets.ini` Docker daemon. Стоит
  добавить `.dockerignore` с `.git`, `.idea`, `.venv`, `config/secrets.ini`.
- `/transcribe` и `/stream` защищены API-ключом.
- `/prompt` сейчас не защищен API-ключом на сервере. Если сервис доступен не
  только доверенным контейнерам/хостам, это надо исправить.
- `whisper_dict.save_prompt()` в `LocalRAGagent0.1` не отправляет `X-API-Key`.
  Если включить защиту `PUT /prompt`, клиент тоже нужно обновить.
- REST `/transcribe` не ресемплит аудио, а требует готовые `16 kHz`. WebSocket
  клиент из `LocalRAGagent0.1` ресемплит на клиентской стороне.
- `MAX_CONCURRENT=8` может быть агрессивным для `large-v3` на одной GPU. Если
  появятся OOM или latency spikes, начать с `MAX_CONCURRENT=1..2`.
- `root_path="/whisper"` полезен для reverse proxy, но не является route-prefix
  при прямом доступе. Это важно учитывать в nginx.

## Открытые вопросы

1. Почему OpenVPN-адрес `172.16.4.68` недоступен, если он должен вести на этот
   AI-сервер? Рабочий адрес сервера при проверке: `172.16.0.16`.
2. Нужно ли защищать `GET /prompt` и `PUT /prompt` тем же `X-API-Key`, что и
   `/transcribe`?
3. Должен ли канонический адрес для внешних клиентов оставаться
   `172.16.0.16:8001`, или его лучше заменить на DNS-имя AI-сервера?
4. Планируется ли запускать этот проект напрямую на хосте без Docker? Если да,
   стоит унифицировать путь к `secrets.ini`.
