# Streaming TTS + Offline STT Pipeline

Микросервисная архитектура для синтеза и распознавания речи с потоковой передачей данных. Проект реализует три независимых сервиса: TTS (Text-to-Speech), ASR (Automatic Speech Recognition) и Gateway для оркестрации запросов.

## 🎯 Возможности

- **Потоковый синтез речи (TTS)** — генерация аудио с передачей данных по мере готовности (chunked streaming)
- **Распознавание речи (STT/ASR)** — транскрибация аудиофайлов в текст с временными метками
- **Echo pipeline** — полный цикл: аудио → текст → синтез → аудио
- **HTTP API** — гибкие интерфейсы для разных сценариев использования
- **Docker Compose** — изолированные контейнеры с единой сетью и персистентным хранилищем моделей

## 🏗️ Архитектура

```
┌──────────┐
│  Client  │
└────┬─────┘
     │
     ▼
┌─────────────────┐
│    Gateway      │ :8000
│  (Orchestrator) │
└────┬────────┬───┘
     │        │
     ▼        ▼
┌─────────┐ ┌─────────┐
│   TTS   │ │   ASR   │
│ :8082   │ │ :8081   │
└─────────┘ └─────────┘
```

### Компоненты

- **TTS Service** — синтез речи на базе Silero TTS (PyTorch)
- **ASR Service** — распознавание речи через Vosk (английский язык)
- **Gateway** — единая точка входа, проксирование и композиция сервисов

## 🚀 Быстрый старт

### Требования

- Docker Desktop или Docker Engine + Docker Compose
- 4+ GB RAM (для загрузки и работы моделей)
- Доступ к интернету (для первой загрузки моделей)

### Установка и запуск

1. **Клонируйте репозиторий**

```bash
git clone <repository-url>
cd streaming-tts-stt
```

2. **Создайте конфигурационный файл .env**

Содержимое `.env`:
```env
ASR_MODEL_SIZE=base.en
# Maximum audio duration in seconds
MAX_AUDIO_DURATION=15
GATEWAY_PORT=8000
TTS_SERVICE_URL=http://tts:8082
ASR_SERVICE_URL=http://asr:8081
# Request timeout in seconds
REQUEST_TIMEOUT=60
```

3. **Запустите сервисы**

```bash
docker-compose up --build
```

При первом запуске произойдёт автоматическая загрузка моделей:
- Silero TTS (English v3) — ~50 MB
- Vosk Small English — ~40 MB


## 📡 API Documentation

### 1. TTS — Синтез речи (через Gateway)

#### HTTP POST `/api/tts`

Генерирует аудио из текста с потоковой отдачей PCM данных.

**Запрос:**
```bash
curl -X POST http://localhost:8000/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' \
  --output speech.pcm
```

**Ответ:**
- `Content-Type: application/octet-stream`
- Формат: PCM s16le, 16kHz, mono
- Заголовки: `X-Audio-Format`, `X-Sample-Rate`, `X-Channels`

#### Прямой доступ к TTS (опционально)

```bash
# PCM формат
curl -X POST http://localhost:8082/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' \
  --output speech.pcm

# WAV формат (можно сразу воспроизвести)
curl -X POST http://localhost:8082/api/tts/wav \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' \
  --output speech.wav
```

### 2. ASR — Распознавание речи

#### HTTP POST `/api/stt/bytes`

Транскрибирует сырые PCM байты в текст.

**Запрос:**
```bash
curl -X POST "http://localhost:8081/api/stt/bytes?sr=16000&ch=1&fmt=s16le" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.pcm
```

**Query параметры:**
- `sr` — sample rate (8000, 16000, 22050, 44100, 48000)
- `ch` — количество каналов (1 или 2)
- `fmt` — формат данных (s16le, s32le, float32)
- `lang` — язык (по умолчанию "en")

**Ответ:**
```json
{
  "text": "hello world this is a test",
  "segments": [
    {"start_ms": 0, "end_ms": 500, "text": "hello"},
    {"start_ms": 500, "end_ms": 1200, "text": "world"}
  ],
  "language": "en",
  "duration_seconds": 3.5
}
```

**Ограничения:**
- Максимальная длительность аудио: 15 секунд
- Только английский язык

### 3. Echo Pipeline — ASR → TTS

#### HTTP POST `/api/echo-bytes`

Полный цикл: распознаёт речь из аудио и синтезирует её обратно.

**Запрос:**
```bash
curl -X POST "http://localhost:8000/api/echo-bytes?sr=16000&ch=1&fmt=s16le" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @input.pcm \
  --output output.pcm
```

**Процесс:**
1. Получает PCM байты от клиента
2. Отправляет в ASR для распознавания
3. Полученный текст передаёт в TTS
4. Стримит синтезированное аудио обратно клиенту

**Ответ:**
- `Content-Type: application/octet-stream`
- Заголовок `X-Recognized-Text` содержит распознанный текст
- Тело: PCM поток (16kHz, mono, s16le)

## 🧪 Тестовые клиенты

В директории `client/` находятся готовые Python скрипты для тестирования.

### 1. TTS клиент

Отправляет текст на синтез и сохраняет результат в WAV:

```bash
cd client
python stream_tts.py
```

Результат:
```
Отправка запроса на http://localhost:8000/api/tts...
Текст: Hello world, this is a streaming TTS test.

Получение аудио-чанков:
Фрейм 1: 4096 байт, время: 1730000000.123
Фрейм 2: 4096 байт, время: 1730000000.145
...

✓ Аудио сохранено в input.wav
Всего фреймов: 42, размер: 168960 байт
```

### 2. Echo клиент

Читает WAV, отправляет на распознавание и синтез, сохраняет результат:

```bash
cd client
python echo_bytes.py
```

Результат:
```
Входной файл: input.wav
Параметры: 1ch, 16000Hz, 16bit
Размер PCM: 168960 байт

Отправка на http://localhost:8000/api/echo-bytes...

Получение ответа:
📝 Распознанный текст: hello world this is a streaming tts test

Фрейм 1: 4096 байт, время: 1730000001.234
Фрейм 2: 4096 байт, время: 1730000001.256
...

✓ Аудио сохранено в out_echo.wav
Всего фреймов: 38, размер: 155648 байт
```

### Docker volumes

Проект использует именованные тома для персистентного хранения:

- `models_tts` — веса Silero TTS модели
- `models_asr` — веса Vosk модели
- `speech_logs` — логи всех сервисов

Просмотр логов:
```bash
# Все сервисы
docker-compose logs -f

# Конкретный сервис
docker-compose logs -f gateway
docker-compose logs -f tts
docker-compose logs -f asr
```



## 🧰 Технологический стек

### Backend
- **Python 3.10** — основной язык разработки
- **FastAPI** — асинхронный веб-фреймворк
- **Uvicorn** — ASGI сервер

### ML Models
- **Silero TTS** (PyTorch) — быстрый русско-английский синтез речи
- **Vosk** — легковесное оффлайн распознавание речи

### Infrastructure
- **Docker & Docker Compose** — контейнеризация и оркестрация
- **httpx** — асинхронный HTTP клиент для межсервисного взаимодействия


**Создано с ❤️ для демонстрации навыков работы с ML, Docker и микросервисной архитектурой**
