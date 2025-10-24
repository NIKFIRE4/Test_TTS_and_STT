#!/usr/bin/env python3
"""
Stream TTS Client
Connects to Gateway WebSocket, sends text, receives PCM chunks and saves to WAV
"""
import asyncio
import json
import sys
import time
from pathlib import Path
import wave

import websockets


GATEWAY_URL = "ws://localhost:8082/ws/tts"
OUTPUT_FILE = "out.wav"
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes


async def stream_tts(text: str, output_path: str):
    """
    Connect to TTS WebSocket, send text, and receive streaming audio
    """
    print(f"[INFO] Connecting to {GATEWAY_URL}")
    print(f"[INFO] Text: {text}")
    
    audio_chunks = []
    chunk_times = []
    start_time = time.time()
    
    try:
        async with websockets.connect(GATEWAY_URL) as websocket:
            # Send text
            message = json.dumps({"text": text})
            await websocket.send(message)
            print(f"[INFO] Sent text at t={time.time() - start_time:.3f}s")
            
            # Receive chunks
            chunk_count = 0
            while True:
                try:
                    message = await websocket.recv()
                    
                    if isinstance(message, bytes):
                        # Binary audio chunk
                        chunk_count += 1
                        chunk_time = time.time() - start_time
                        audio_chunks.append(message)
                        chunk_times.append(chunk_time)
                        
                        print(f"[CHUNK {chunk_count}] Received {len(message)} bytes at t={chunk_time:.3f}s")
                    
                    else:
                        # Text message (JSON)
                        data = json.loads(message)
                        
                        if data.get("type") == "end":
                            print(f"[INFO] Stream ended at t={time.time() - start_time:.3f}s")
                            break
                        
                        elif data.get("type") == "error":
                            print(f"[ERROR] {data.get('message')}")
                            return
                        
                except websockets.exceptions.ConnectionClosed:
                    print("[INFO] Connection closed")
                    break
        
        # Save to WAV
        if audio_chunks:
            total_bytes = sum(len(chunk) for chunk in audio_chunks)
            print(f"\n[INFO] Received {chunk_count} chunks, {total_bytes} bytes total")
            
            # Write WAV file
            with wave.open(output_path, 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(SAMPLE_WIDTH)
                wav_file.setframerate(SAMPLE_RATE)
                
                for chunk in audio_chunks:
                    wav_file.writeframes(chunk)
            
            duration = total_bytes / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)
            print(f"[INFO] Saved to {output_path} (duration: {duration:.2f}s)")
            
            # Statistics
            if len(chunk_times) > 1:
                intervals = [chunk_times[i] - chunk_times[i-1] for i in range(1, len(chunk_times))]
                avg_interval = sum(intervals) / len(intervals)
                print(f"[STATS] Average chunk interval: {avg_interval:.3f}s")
                print(f"[STATS] Total time: {chunk_times[-1]:.3f}s")
        
        else:
            print("[WARNING] No audio received")
    
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def main():
    if len(sys.argv) < 2:
        print("Usage: python stream_tts.py \"Your text here\" [output.wav]")
        print("Example: python stream_tts.py \"Hello world\" out.wav")
        sys.exit(1)
    
    text = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_FILE
    
    # Run async
    asyncio.run(stream_tts(text, output_path))


if __name__ == "__main__":
    main()