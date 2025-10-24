"""
ASR Service - Automatic Speech Recognition (Vosk version for Windows)
Uses Vosk for easier Windows deployment
"""
import os
import logging
import sys
from typing import List, Dict
import json
import wave
import io

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from vosk import Model, KaldiRecognizer
import uvicorn

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL,
    format='{"time":"%(asctime)s", "level":"%(levelname)s", "service":"asr", "message":"%(message)s"}',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/var/log/app/asr.log"),
    ],
)
logger = logging.getLogger(__name__)

# Configuration
PORT = int(os.getenv("ASR_PORT", "8081"))
MODEL_PATH = os.getenv("MODEL_PATH", "/opt/models/vosk-model-en")
MAX_DURATION = int(os.getenv("MAX_AUDIO_DURATION", "15"))

app = FastAPI(title="ASR Service (Vosk)", version="1.0.0")

# Global model instance
asr_model = None


def load_model():
    """Load Vosk model"""
    global asr_model
    try:
        logger.info("loading_vosk_model", extra={"model_path": MODEL_PATH})
        
        # Check if model exists, if not download
        if not os.path.exists(MODEL_PATH) or not os.path.exists(os.path.join(MODEL_PATH, "am")):
            logger.info("downloading_vosk_model")
            import urllib.request
            import zipfile
            import shutil
            
            os.makedirs(MODEL_PATH, exist_ok=True)
            
            # Download small English model
            url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
            zip_path = "/tmp/vosk-model.zip"
            
            logger.info("downloading_from_url", extra={"url": url})
            urllib.request.urlretrieve(url, zip_path)
            
            logger.info("extracting_model")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("/tmp")
            
            # Move contents to MODEL_PATH
            extracted = "/tmp/vosk-model-small-en-us-0.15"
            if os.path.exists(extracted):
                for item in os.listdir(extracted):
                    src = os.path.join(extracted, item)
                    dst = os.path.join(MODEL_PATH, item)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)
                logger.info("model_extracted_successfully")
        
        model = Model(MODEL_PATH)
        asr_model = model
        
        logger.info("vosk_model_loaded")
    except Exception as e:
        logger.error("failed_to_load_model", extra={"error": str(e)})
        raise


def bytes_to_audio(
    audio_bytes: bytes,
    sample_rate: int,
    channels: int,
    dtype: str = "int16"
) -> np.ndarray:
    """Convert raw PCM bytes to numpy array"""
    
    dtype_map = {
        "int16": np.int16,
        "s16le": np.int16,
        "int32": np.int32,
        "float32": np.float32,
    }
    
    np_dtype = dtype_map.get(dtype, np.int16)
    audio = np.frombuffer(audio_bytes, dtype=np_dtype)
    
    # Convert to int16 if needed
    if np_dtype == np.float32:
        audio = (audio * 32767).astype(np.int16)
    elif np_dtype == np.int32:
        audio = (audio / 65536).astype(np.int16)
    
    # Handle stereo to mono
    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
    
    return audio


def transcribe_audio(
    audio: np.ndarray,
    sample_rate: int,
    language: str = "en"
) -> Dict:
    """
    Transcribe audio using Vosk
    Returns dict with text and segments
    """
    try:
        duration = len(audio) / sample_rate
        logger.info("transcribing_audio", extra={
            "duration_sec": duration,
            "sample_rate": sample_rate
        })
        
        if duration > MAX_DURATION:
            raise ValueError(f"Audio duration {duration:.1f}s exceeds maximum {MAX_DURATION}s")
        
        # Create recognizer
        rec = KaldiRecognizer(asr_model, sample_rate)
        rec.SetWords(True)  # Enable word-level timestamps
        
        # Convert audio to bytes for Vosk
        audio_bytes = audio.tobytes()
        
        # Process audio
        rec.AcceptWaveform(audio_bytes)
        
        # Get final result
        result = json.loads(rec.FinalResult())
        
        full_text = result.get("text", "")
        
        # Extract segments with timestamps if available
        segments = []
        if "result" in result:
            for word_info in result["result"]:
                segments.append({
                    "start_ms": int(word_info.get("start", 0) * 1000),
                    "end_ms": int(word_info.get("end", 0) * 1000),
                    "text": word_info.get("word", ""),
                })
        
        output = {
            "text": full_text,
            "segments": segments,
            "language": "en",
            "duration_seconds": duration,
        }
        
        logger.info("transcription_completed", extra={
            "text_length": len(full_text),
            "num_segments": len(segments)
        })
        
        return output
        
    except Exception as e:
        logger.error("transcription_failed", extra={"error": str(e)})
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    load_model()



@app.post("/api/stt/bytes")
async def stt_from_bytes(request: Request):
    """
    Transcribe audio from raw PCM bytes
    Query params: sr (sample_rate), ch (channels), lang (optional)
    Body: application/octet-stream (raw PCM bytes)
    """
    sample_rate = int(request.query_params.get("sr", "16000"))
    channels = int(request.query_params.get("ch", "1"))
    language = request.query_params.get("lang", "en")
    fmt = request.query_params.get("fmt", "s16le")
    
    if sample_rate not in [8000, 16000, 22050, 44100, 48000]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sample rate: {sample_rate}"
        )
    
    if channels not in [1, 2]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid channel count: {channels}"
        )
    
    try:
        audio_bytes = await request.body()
        
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio data")
        
        logger.info("received_audio", extra={
            "size_bytes": len(audio_bytes),
            "sample_rate": sample_rate,
            "channels": channels
        })
        
        audio = bytes_to_audio(audio_bytes, sample_rate, channels, fmt)
        result = transcribe_audio(audio, sample_rate, language)
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("stt_error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_config=None,
    )