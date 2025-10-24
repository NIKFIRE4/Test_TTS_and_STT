import requests
import time
import wave

def stream_tts_http(text: str, output_file: str = "input.wav"):
    """
    Отправляет текст на TTS через HTTP и сохраняет chunked-поток PCM в WAV файл.
    """
    url = "http://localhost:8000/api/tts"
    headers = {"Content-Type": "application/json"}
    data = {"text": text}
    
    print(f"Отправка запроса на {url}...")
    print(f"Текст: {text}")
    
    try:
        response = requests.post(url, json=data, stream=True, headers=headers)
        response.raise_for_status()
        
        pcm_chunks = []
        chunk_count = 0
        
        print("\nПолучение аудио-чанков:")
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                chunk_count += 1
                pcm_chunks.append(chunk)
                print(f"Фрейм {chunk_count}: {len(chunk)} байт, время: {time.time():.3f}")
        
        # Сохранение в WAV
        pcm_data = b''.join(pcm_chunks)
        with wave.open(output_file, 'wb') as wav_file:
            wav_file.setnchannels(1)  # моно
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)  # 16kHz
            wav_file.writeframes(pcm_data)
        
        print(f"\n✓ Аудио сохранено в {output_file}")
        print(f"Всего фреймов: {chunk_count}, размер: {len(pcm_data)} байт")
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Ошибка запроса: {e}")

if __name__ == "__main__":
    stream_tts_http("Hello world, this is a streaming TTS test.")
