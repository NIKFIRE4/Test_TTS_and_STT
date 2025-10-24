import os
import json
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import torch
import torchaudio

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s", "level":"%(levelname)s", "service":"tts", "message":"%(message)s"}',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Конфигурация
TTS_PORT = int(os.getenv("TTS_PORT", "8082"))

# Глобальное хранилище для модели
MODEL_STATE = {
    "model": None,
    "device": None,
    "sample_rate": 48000,
    "speaker": "en_0"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("loading_tts_model")
    try:
        # Определяем устройство
        device = torch.device('cpu')
        logger.info(f"using_device: {device}")
        
        # Загружаем модель Silero TTS
        logger.info("downloading_model_from_torch_hub")
        model, example_text = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='en',
            speaker='v3_en',
            verbose=True
        )
        
        logger.info("model_downloaded_successfully")
        
        # Сохраняем в глобальный словарь СРАЗУ после загрузки
        MODEL_STATE["model"] = model
        MODEL_STATE["device"] = device
        
        logger.info("tts_model_loaded_successfully")
        logger.info(f"model_state_check: model_is_none={MODEL_STATE['model'] is None}")
        logger.info("startup_complete")
        
    except Exception as e:
        logger.error(f"model_load_error: {str(e)}", exc_info=True)
        MODEL_STATE["model"] = None
        logger.warning("continuing_without_model")
    
    yield
    
    logger.info("tts_shutting_down")
    # Очистка ресурсов
    if MODEL_STATE["model"] is not None:
        del MODEL_STATE["model"]
        MODEL_STATE["model"] = None

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    """Health check endpoint"""
    model_loaded = MODEL_STATE["model"] is not None
    model_status = "loaded" if model_loaded else "not_loaded"
    logger.info(f"health_check: model_status={model_status}")
    return {
        "status": "healthy" if model_loaded else "degraded",
        "service": "tts",
        "model_loaded": model_loaded
    }

async def generate_audio_stream(text: str, chunk_size: int = 4096):
    """
    Генератор для стриминга аудио данных
    """
    try:
        # Проверяем наличие модели
        if MODEL_STATE["model"] is None:
            logger.error("model_not_loaded")
            raise Exception("Model not loaded")
        
        logger.info(f"synthesizing_text: {text[:50]}...")
        
        # Получаем параметры из глобального состояния
        model = MODEL_STATE["model"]
        sample_rate = MODEL_STATE["sample_rate"]
        speaker = MODEL_STATE["speaker"]
        
        # Генерируем аудио через Silero TTS
        logger.info("calling_apply_tts")
        audio = model.apply_tts(
            text=text,
            speaker=speaker,
            sample_rate=sample_rate
        )
        
        logger.info(f"audio_tensor_shape: {audio.shape}")
        
        # Ресемплинг до 16kHz если нужно
        if sample_rate != 16000:
            logger.info("resampling_to_16khz")
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            )
            audio = resampler(audio)
        
        # Нормализация и конвертация в int16
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = (audio * 32767).clamp(-32768, 32767).to(torch.int16)
        
        # Конвертируем в байты
        audio_bytes = audio.cpu().numpy().tobytes()
        
        logger.info(f"audio_generated_bytes: {len(audio_bytes)}")
        
        # Отправляем аудио чанками
        total_chunks = (len(audio_bytes) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            yield chunk
            # Небольшая задержка для имитации стриминга
            await asyncio.sleep(0.01)
        
        logger.info(f"sent_audio_chunks: {total_chunks}")
        logger.info("tts_completed")
        
    except Exception as e:
        logger.error(f"synthesis_error: {str(e)}", exc_info=True)
        raise

async def generate_wav_stream(text: str, chunk_size: int = 4096):
    """
    Генератор для стриминга WAV аудио данных
    """
    import io
    import struct
    
    try:
        # Проверяем наличие модели
        if MODEL_STATE["model"] is None:
            logger.error("model_not_loaded")
            raise Exception("Model not loaded")
        
        logger.info(f"synthesizing_text_wav: {text[:50]}...")
        
        # Получаем параметры из глобального состояния
        model = MODEL_STATE["model"]
        sample_rate = MODEL_STATE["sample_rate"]
        speaker = MODEL_STATE["speaker"]
        
        # Генерируем аудио через Silero TTS
        logger.info("calling_apply_tts")
        audio = model.apply_tts(
            text=text,
            speaker=speaker,
            sample_rate=sample_rate
        )
        
        logger.info(f"audio_tensor_shape: {audio.shape}")
        
        # Ресемплинг до 16kHz если нужно
        if sample_rate != 16000:
            logger.info("resampling_to_16khz")
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            )
            audio = resampler(audio)
        
        # Нормализация и конвертация в int16
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = (audio * 32767).clamp(-32768, 32767).to(torch.int16)
        
        # Конвертируем в байты
        audio_bytes = audio.cpu().numpy().tobytes()
        
        logger.info(f"audio_generated_bytes: {len(audio_bytes)}")
        
        # Параметры WAV
        num_channels = 1
        sample_width = 2  # 16-bit
        frame_rate = 16000
        num_frames = len(audio_bytes) // sample_width
        
        # Создаем WAV заголовок
        wav_header = io.BytesIO()
        
        # RIFF заголовок
        wav_header.write(b'RIFF')
        wav_header.write(struct.pack('<I', 36 + len(audio_bytes)))  # Размер файла - 8
        wav_header.write(b'WAVE')
        
        # fmt подчанк
        wav_header.write(b'fmt ')
        wav_header.write(struct.pack('<I', 16))  # Размер fmt чанка
        wav_header.write(struct.pack('<H', 1))   # Аудио формат (1 = PCM)
        wav_header.write(struct.pack('<H', num_channels))
        wav_header.write(struct.pack('<I', frame_rate))
        wav_header.write(struct.pack('<I', frame_rate * num_channels * sample_width))  # Byte rate
        wav_header.write(struct.pack('<H', num_channels * sample_width))  # Block align
        wav_header.write(struct.pack('<H', sample_width * 8))  # Bits per sample
        
        # data подчанк
        wav_header.write(b'data')
        wav_header.write(struct.pack('<I', len(audio_bytes)))
        
        # Отправляем WAV заголовок
        header_bytes = wav_header.getvalue()
        yield header_bytes
        logger.info(f"sent_wav_header: {len(header_bytes)} bytes")
        
        # Отправляем аудио данные чанками
        total_chunks = (len(audio_bytes) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            yield chunk
            # Небольшая задержка для имитации стриминга
            await asyncio.sleep(0.01)
        
        logger.info(f"sent_audio_chunks: {total_chunks}")
        logger.info("tts_wav_completed")
        
    except Exception as e:
        logger.error(f"synthesis_error: {str(e)}", exc_info=True)
        raise

@app.post("/api/tts")
async def tts_endpoint(request: Request):
    """
    HTTP POST endpoint для TTS стриминга
    Принимает: JSON {"text": "..."}
    Возвращает: chunked response с бинарными PCM фреймами (16kHz, mono, s16le)
    """
    try:
        # Получаем JSON из тела запроса
        body = await request.json()
        text = body.get("text", "")
        
        if not text:
            logger.error("no_text_provided")
            return {"error": "No text provided"}, 400
        
        logger.info("received_tts_request")
        
        # Возвращаем StreamingResponse с генератором
        return StreamingResponse(
            generate_audio_stream(text),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": "attachment; filename=speech.pcm",
                "X-Audio-Format": "pcm_s16le",
                "X-Sample-Rate": "16000",
                "X-Channels": "1"
            }
        )
        
    except Exception as e:
        logger.error(f"tts_endpoint_error: {str(e)}", exc_info=True)
        return {"error": str(e)}, 500

@app.post("/api/tts/wav")
async def tts_wav_endpoint(request: Request):
    """
    HTTP POST endpoint для TTS стриминга в формате WAV
    Принимает: JSON {"text": "..."}
    Возвращает: chunked response с WAV файлом (16kHz, mono, 16-bit)
    Можно сразу прослушать в браузере или Postman
    """
    try:
        # Получаем JSON из тела запроса
        body = await request.json()
        text = body.get("text", "")
        
        if not text:
            logger.error("no_text_provided")
            return {"error": "No text provided"}, 400
        
        logger.info("received_tts_wav_request")
        
        # Возвращаем StreamingResponse с WAV генератором
        return StreamingResponse(
            generate_wav_stream(text),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
            }
        )
        
    except Exception as e:
        logger.error(f"tts_wav_endpoint_error: {str(e)}", exc_info=True)
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=TTS_PORT)