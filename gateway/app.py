import os
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import httpx


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s", "level":"%(levelname)s", "service":"gateway", "message":"%(message)s"}',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Конфигурация из переменных окружения
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://localhost:8082")
ASR_SERVICE_URL = os.getenv("ASR_SERVICE_URL", "http://localhost:8081")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8000"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("gateway_starting", extra={
        "tts_service_url": TTS_SERVICE_URL,
        "asr_service_url": ASR_SERVICE_URL,
        "port": GATEWAY_PORT
    })
    yield
    logger.info("gateway_shutting_down")


app = FastAPI(
    title="Gateway Service",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "gateway",
        "tts_service_url": TTS_SERVICE_URL,
        "asr_service_url": ASR_SERVICE_URL
    }


@app.get("/health/full")
async def full_health_check():
    """
    Полная проверка здоровья - проверяет доступность всех сервисов
    """
    services = {
        "gateway": "healthy",
        "tts": "unknown",
        "asr": "unknown"
    }
    
    # Проверяем TTS
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            tts_response = await client.get(f"{TTS_SERVICE_URL}/health")
            if tts_response.status_code == 200:
                tts_data = tts_response.json()
                services["tts"] = "healthy" if tts_data.get("model_loaded") else "degraded"
            else:
                services["tts"] = "error"
    except Exception as e:
        services["tts"] = f"unreachable: {str(e)}"
        logger.warning(f"tts_health_check_failed: {e}")
    
    # Проверяем ASR
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            asr_response = await client.get(f"{ASR_SERVICE_URL}/health")
            if asr_response.status_code == 200:
                asr_data = asr_response.json()
                services["asr"] = "healthy" if asr_data.get("model_loaded") else "degraded"
            else:
                services["asr"] = "error"
    except Exception as e:
        services["asr"] = f"unreachable: {str(e)}"
        logger.warning(f"asr_health_check_failed: {e}")
    
    all_healthy = all(v == "healthy" for v in services.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "services": services
    }


@app.post("/api/echo-bytes")
async def echo_bytes(request: Request):
    """
    HTTP endpoint для ASR -> TTS pipeline (echo)
    
    Вход:
      - Body: raw PCM bytes (application/octet-stream)
      - Query params: sr (sample_rate), ch (channels), fmt (format)
    
    Процесс:
      1. Получает сырые PCM байты
      2. Отправляет в ASR для распознавания
      3. Отправляет распознанный текст в TTS
      4. Возвращает chunked PCM поток
    
    Выход:
      - Streaming response с PCM данными
      - Header X-Recognized-Text с распознанным текстом
    """
    # Получаем параметры аудио
    sr = int(request.query_params.get("sr", "16000"))
    ch = int(request.query_params.get("ch", "1"))
    fmt = request.query_params.get("fmt", "s16le")
    lang = request.query_params.get("lang", "en")
    
    logger.info("echo_bytes_request", extra={
        "sr": sr,
        "ch": ch,
        "fmt": fmt,
        "lang": lang
    })
    
    try:
        # Шаг 1: Читаем входящие аудио байты
        audio_bytes = await request.body()
        
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio data")
        
        logger.info("received_audio", extra={"size_bytes": len(audio_bytes)})
        
        # Шаг 2: Отправляем в ASR сервис для распознавания
        asr_url = f"{ASR_SERVICE_URL}/api/stt/bytes"
        
        logger.info("calling_asr_service", extra={"url": asr_url})
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            asr_response = await client.post(
                asr_url,
                params={
                    "sr": sr,
                    "ch": ch,
                    "fmt": fmt,
                    "lang": lang
                },
                content=audio_bytes,
                headers={"Content-Type": "application/octet-stream"}
            )
            
            if asr_response.status_code != 200:
                error_detail = f"ASR service error: HTTP {asr_response.status_code}"
                try:
                    error_data = asr_response.json()
                    error_detail += f" - {error_data.get('detail', '')}"
                except:
                    error_detail += f" - {asr_response.text[:200]}"
                
                logger.error("asr_service_error", extra={"detail": error_detail})
                raise HTTPException(status_code=500, detail=error_detail)
            
            asr_result = asr_response.json()
            recognized_text = asr_result.get("text", "")
            segments = asr_result.get("segments", [])
            
            logger.info("asr_completed", extra={
                "recognized_text": recognized_text,
                "num_segments": len(segments),
                "duration": asr_result.get("duration_seconds", 0)
            })
            
            if not recognized_text:
                raise HTTPException(status_code=400, detail="No text recognized from audio")
        
        # Шаг 3: Генератор для стриминга TTS ответа
        async def tts_stream_generator():
            """
            Генератор для chunked streaming TTS ответа
            """
            tts_url = f"{TTS_SERVICE_URL}/api/tts"
            
            logger.info("calling_tts_service", extra={
                "url": tts_url,
                "text": recognized_text
            })
            
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    # Отправляем распознанный текст в TTS
                    async with client.stream(
                        "POST",
                        tts_url,
                        json={"text": recognized_text},
                        headers={"Content-Type": "application/json"}
                    ) as tts_response:
                        
                        if tts_response.status_code != 200:
                            error_msg = f"TTS service error: HTTP {tts_response.status_code}"
                            logger.error("tts_service_error", extra={"detail": error_msg})
                            raise Exception(error_msg)
                        
                        logger.info("tts_streaming_started")
                        
                        chunk_count = 0
                        total_bytes = 0
                        
                        # Стримим PCM фреймы от TTS к клиенту
                        async for chunk in tts_response.aiter_bytes(chunk_size=4096):
                            if chunk:
                                chunk_count += 1
                                total_bytes += len(chunk)
                                yield chunk
                        
                        logger.info("tts_streaming_completed", extra={
                            "chunks": chunk_count,
                            "total_bytes": total_bytes
                        })
                        
            except Exception as e:
                logger.error("tts_stream_error", extra={"error": str(e)})
                raise
        
        # Шаг 4: Возвращаем streaming response
        return StreamingResponse(
            tts_stream_generator(),
            media_type="application/octet-stream",
            headers={
                "X-Recognized-Text": recognized_text,
                "X-Audio-Format": "pcm_s16le",
                "X-Sample-Rate": "16000",
                "X-Channels": "1",
                "Content-Disposition": "attachment; filename=echo.pcm"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("echo_bytes_error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/api/tts")
async def gateway_tts(request: Request):
    """
    Прокси-эндпоинт для TTS (опционально)
    Принимает JSON {"text": "..."} и возвращает PCM stream
    """
    try:
        body = await request.json()
        text = body.get("text", "")
        
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
        
        logger.info("gateway_tts_request", extra={"text": text[:100]})
        
        async def tts_proxy_generator():
            tts_url = f"{TTS_SERVICE_URL}/api/tts"
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    tts_url,
                    json={"text": text}
                ) as response:
                    if response.status_code != 200:
                        raise HTTPException(
                            status_code=response.status_code,
                            detail="TTS service error"
                        )
                    
                    async for chunk in response.aiter_bytes(chunk_size=4096):
                        if chunk:
                            yield chunk
        
        return StreamingResponse(
            tts_proxy_generator(),
            media_type="application/octet-stream",
            headers={
                "X-Audio-Format": "pcm_s16le",
                "X-Sample-Rate": "16000",
                "X-Channels": "1"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("gateway_tts_error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stt")
async def gateway_stt(request: Request):
    """
    Прокси-эндпоинт для ASR (опционально)
    Принимает raw PCM bytes и возвращает JSON с распознанным текстом
    """
    try:
        sr = int(request.query_params.get("sr", "16000"))
        ch = int(request.query_params.get("ch", "1"))
        fmt = request.query_params.get("fmt", "s16le")
        lang = request.query_params.get("lang", "en")
        
        audio_bytes = await request.body()
        
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio data")
        
        logger.info("gateway_stt_request", extra={"size_bytes": len(audio_bytes)})
        
        asr_url = f"{ASR_SERVICE_URL}/api/stt/bytes"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                asr_url,
                params={"sr": sr, "ch": ch, "fmt": fmt, "lang": lang},
                content=audio_bytes,
                headers={"Content-Type": "application/octet-stream"}
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="ASR service error"
                )
            
            return response.json()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("gateway_stt_error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=GATEWAY_PORT,
        log_config=None
    )
