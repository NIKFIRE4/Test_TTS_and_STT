"""
Gateway Service - Entry point for client requests
Proxies and orchestrates TTS and ASR services
"""
import os
import logging
import sys
import json
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import StreamingResponse
import uvicorn
import websockets

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL,
    format='{"time":"%(asctime)s", "level":"%(levelname)s", "service":"gateway", "message":"%(message)s"}',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/var/log/app/gateway.log"),
    ],
)
logger = logging.getLogger(__name__)

# Configuration
PORT = int(os.getenv("GATEWAY_PORT", "8000"))
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://tts:8082")
ASR_SERVICE_URL = os.getenv("ASR_SERVICE_URL", "http://asr:8081")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))

app = FastAPI(title="Gateway Service", version="1.0.0")


async def stream_from_tts_ws(text: str) -> AsyncGenerator[bytes, None]:
    """
    Connect to TTS WebSocket and stream audio chunks
    """
    tts_ws_url = TTS_SERVICE_URL.replace("http://", "ws://") + "/ws/tts"
    
    try:
        logger.info("connecting_to_tts", extra={"url": tts_ws_url})
        
        async with websockets.connect(tts_ws_url) as websocket:
            # Send text
            await websocket.send(json.dumps({"text": text}))
            logger.info("sent_text_to_tts", extra={"text_length": len(text)})
            
            # Receive and stream chunks
            chunk_count = 0
            while True:
                try:
                    message = await websocket.recv()
                    
                    # Check if it's binary (audio) or text (end marker)
                    if isinstance(message, bytes):
                        chunk_count += 1
                        yield message
                    else:
                        # Text message (JSON)
                        data = json.loads(message)
                        if data.get("type") == "end":
                            logger.info("tts_stream_ended", extra={"chunks_received": chunk_count})
                            break
                        elif data.get("type") == "error":
                            raise Exception(f"TTS error: {data.get('message')}")
                            
                except websockets.exceptions.ConnectionClosed:
                    logger.info("tts_connection_closed")
                    break
                    
    except Exception as e:
        logger.error("tts_streaming_error", extra={"error": str(e)})
        raise


async def call_asr_service(
    audio_bytes: bytes,
    sample_rate: int,
    channels: int,
    fmt: str = "s16le"
) -> dict:
    """
    Call ASR service to transcribe audio
    """
    url = f"{ASR_SERVICE_URL}/api/stt/bytes"
    params = {
        "sr": sample_rate,
        "ch": channels,
        "fmt": fmt,
        "lang": "en",
    }
    
    try:
        logger.info("calling_asr_service", extra={
            "audio_size": len(audio_bytes),
            "sample_rate": sample_rate
        })
        
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                url,
                params=params,
                content=audio_bytes,
                headers={"Content-Type": "application/octet-stream"},
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info("asr_completed", extra={
                "text_length": len(result.get("text", ""))
            })
            
            return result
            
    except httpx.HTTPError as e:
        logger.error("asr_request_failed", extra={"error": str(e)})
        raise HTTPException(status_code=502, detail=f"ASR service error: {str(e)}")
    except Exception as e:
        logger.error("asr_error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    Checks connectivity to downstream services
    """
    status = {"gateway": "healthy", "services": {}}
    
    # Check TTS service
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{TTS_SERVICE_URL}/health")
            status["services"]["tts"] = "healthy" if resp.status_code == 200 else "unhealthy"
    except Exception as e:
        status["services"]["tts"] = f"error: {str(e)}"
    
    # Check ASR service
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{ASR_SERVICE_URL}/health")
            status["services"]["asr"] = "healthy" if resp.status_code == 200 else "unhealthy"
    except Exception as e:
        status["services"]["asr"] = f"error: {str(e)}"
    
    # Overall status
    all_healthy = all(
        v == "healthy" 
        for k, v in status["services"].items()
    )
    
    if not all_healthy:
        return status, 503
    
    return status


@app.websocket("/ws/tts")
async def websocket_gateway_tts(websocket: WebSocket):
    """
    WebSocket endpoint for TTS
    Input: {"text": "..."} or {"segments": [{"text": "..."}, ...]}
    Output: Streaming PCM chunks + {"type": "end"}
    """
    await websocket.accept()
    logger.info("gateway_ws_connected")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                
                # Handle single text
                if "text" in message:
                    text = message["text"]
                    if not text:
                        await websocket.send_json({"type": "error", "message": "Empty text"})
                        continue
                    
                    # Stream from TTS
                    async for chunk in stream_from_tts_ws(text):
                        await websocket.send_bytes(chunk)
                    
                    # Send end marker
                    await websocket.send_json({"type": "end"})
                
                # Handle segments
                elif "segments" in message:
                    segments = message["segments"]
                    
                    for i, segment in enumerate(segments):
                        text = segment.get("text", "")
                        if not text:
                            continue
                        
                        logger.info("processing_segment", extra={"index": i, "total": len(segments)})
                        
                        # Stream from TTS
                        async for chunk in stream_from_tts_ws(text):
                            await websocket.send_bytes(chunk)
                    
                    # Send end marker after all segments
                    await websocket.send_json({"type": "end"})
                
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid message format. Expected 'text' or 'segments'"
                    })
                    
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
            except Exception as e:
                logger.error("gateway_ws_error", extra={"error": str(e)})
                await websocket.send_json({"type": "error", "message": str(e)})
                
    except WebSocketDisconnect:
        logger.info("gateway_ws_disconnected")
    except Exception as e:
        logger.error("gateway_ws_fatal_error", extra={"error": str(e)})


@app.post("/api/echo-bytes")
async def echo_bytes(request: Request):
    """
    Echo endpoint: ASR → TTS pipeline
    1. Receive raw PCM audio
    2. Call ASR to get text/segments
    3. Call TTS with recognized text
    4. Stream back synthesized audio
    
    Query params: sr, ch, fmt (same as ASR)
    """
    # Get parameters
    sample_rate = int(request.query_params.get("sr", "16000"))
    channels = int(request.query_params.get("ch", "1"))
    fmt = request.query_params.get("fmt", "s16le")
    
    try:
        # Read input audio
        audio_bytes = await request.body()
        
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio data")
        
        logger.info("echo_request", extra={
            "audio_size": len(audio_bytes),
            "sample_rate": sample_rate
        })
        
        # Step 1: Transcribe audio
        asr_result = await call_asr_service(audio_bytes, sample_rate, channels, fmt)
        
        recognized_text = asr_result.get("text", "")
        segments = asr_result.get("segments", [])
        
        if not recognized_text:
            raise HTTPException(status_code=400, detail="No speech detected in audio")
        
        logger.info("recognized_text", extra={
            "text": recognized_text,
            "num_segments": len(segments)
        })
        
        # Step 2: Generate TTS stream
        # Use segments if available, otherwise use full text
        async def generate_echo_stream():
            if segments:
                # Stream each segment
                for segment in segments:
                    segment_text = segment.get("text", "")
                    if segment_text:
                        async for chunk in stream_from_tts_ws(segment_text):
                            yield chunk
            else:
                # Stream full text
                async for chunk in stream_from_tts_ws(recognized_text):
                    yield chunk
        
        return StreamingResponse(
            generate_echo_stream(),
            media_type="application/octet-stream",
            headers={
                "X-Sample-Rate": "16000",
                "X-Channels": "1",
                "X-Format": "s16le",
                "X-Recognized-Text": recognized_text,
            },
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("echo_error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_config=None,
    )