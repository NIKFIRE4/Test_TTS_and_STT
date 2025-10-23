"""
TTS Service - Text-to-Speech with streaming PCM output
Uses Silero TTS model for CPU-efficient synthesis
"""
import os
import json
import logging
import sys
from typing import Generator
import asyncio

import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL,
    format='{"time":"%(asctime)s", "level":"%(levelname)s", "service":"tts", "message":"%(message)s"}',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/var/log/app/tts.log"),
    ],
)
logger = logging.getLogger(__name__)

# Configuration
PORT = int(os.getenv("TTS_PORT", "8082"))
MODEL_PATH = os.getenv("MODEL_PATH", "/opt/models")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "4096"))
SAMPLE_RATE = 16000

app = FastAPI(title="TTS Service", version="1.0.0")

# Global model instance
tts_model = None
device = torch.device("cpu")


def load_model():
    """Load Silero TTS model"""
    global tts_model
    
    logger.info("loading_tts_model", extra={"model_path": MODEL_PATH})
    
    # Load Silero TTS model (v3_en)
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="en",
        speaker="v3_en",
        trust_repo=True,
    )
    model = model.to(device)
    
    # Assign to global variable
    tts_model = model
    
    logger.info("tts_model_loaded", extra={
        "device": str(device),
        "model_type": str(type(tts_model)),
        "model_is_none": tts_model is None
    })


def synthesize_audio_stream(text: str) -> Generator[bytes, None, None]:
    """
    Generate audio stream from text
    Yields fixed-size PCM chunks
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    # Check model directly
    if tts_model is None:
        logger.error("model_check_failed", extra={"tts_model_is_none": True})
        raise RuntimeError("TTS model not loaded")
    
    try:
        logger.info("synthesizing_speech", extra={"text_length": len(text)})
        
        # Generate audio tensor
        audio = tts_model.apply_tts(
            text=text.strip(),
            speaker="en_0",
            sample_rate=SAMPLE_RATE,
        )
        
        # Convert to int16 PCM
        audio_np = (audio.cpu().numpy() * 32767).astype(np.int16)
        audio_bytes = audio_np.tobytes()
        
        total_size = len(audio_bytes)
        logger.info("audio_generated", extra={"size_bytes": total_size})
        
        # Yield in fixed chunks
        for i in range(0, total_size, CHUNK_SIZE):
            chunk = audio_bytes[i:i + CHUNK_SIZE]
            yield chunk
            
    except Exception as e:
        logger.error("synthesis_failed", extra={"error": str(e), "error_type": type(e).__name__})
        raise


async def stream_audio_generator(text: str):
    """Async wrapper for audio streaming"""
    for chunk in synthesize_audio_stream(text):
        yield chunk
        # Small delay to simulate streaming behavior
        await asyncio.sleep(0.01)


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    load_model()
    logger.info("startup_complete", extra={"model_loaded": tts_model is not None})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    is_loaded = tts_model is not None
    logger.info("health_check", extra={"model_loaded": is_loaded})
    return {
        "status": "healthy",
        "service": "tts",
        "model_loaded": is_loaded,
    }


@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS
    Input: {"text": "..."}
    Output: Binary PCM chunks + {"type": "end"}
    """
    await websocket.accept()
    logger.info("websocket_connected")
    
    try:
        while True:
            # Receive text message
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                text = message.get("text", "")
                
                if not text:
                    await websocket.send_json({"type": "error", "message": "Empty text"})
                    continue
                
                logger.info("received_tts_request", extra={"text_preview": text[:50]})
                
                # Stream audio chunks
                chunk_count = 0
                async for chunk in stream_audio_generator(text):
                    await websocket.send_bytes(chunk)
                    chunk_count += 1
                
                # Send end marker
                await websocket.send_json({"type": "end"})
                logger.info("streaming_completed", extra={"chunks_sent": chunk_count})
                
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
            except Exception as e:
                logger.error("tts_error", extra={"error": str(e), "error_type": type(e).__name__})
                await websocket.send_json({"type": "error", "message": f"TTS error: {str(e)}"})
                
    except WebSocketDisconnect:
        logger.info("websocket_disconnected")
    except Exception as e:
        logger.error("websocket_error", extra={"error": str(e)})


@app.post("/api/tts")
async def http_tts(request: dict):
    """
    HTTP endpoint for streaming TTS
    Input: {"text": "..."}
    Output: Chunked PCM stream (application/octet-stream)
    """
    text = request.get("text", "")
    
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    logger.info("http_tts_request", extra={"text_length": len(text)})
    
    return StreamingResponse(
        stream_audio_generator(text),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Channels": "1",
            "X-Format": "s16le",
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_config=None,  # Use our custom logging
    )