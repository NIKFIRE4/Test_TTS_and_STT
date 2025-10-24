import requests
import wave
import time

def echo_bytes_http(input_file: str = "input.wav", output_file: str = "out_echo.wav"):
    """
    Читает WAV файл, отправляет PCM байты на /api/echo-bytes,
    получает распознанный текст и синтезированное аудио обратно.
    """
    # Чтение WAV и извлечение PCM данных
    with wave.open(input_file, 'rb') as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        pcm_data = wav_file.readframes(wav_file.getnframes())
    
    print(f"Входной файл: {input_file}")
    print(f"Параметры: {channels}ch, {framerate}Hz, {sample_width*8}bit")
    print(f"Размер PCM: {len(pcm_data)} байт")
    
    # Определение формата
    fmt_map = {1: "s8", 2: "s16le", 4: "s32le"}
    fmt = fmt_map.get(sample_width, "s16le")
    
    # Отправка на /api/echo-bytes
    url = f"http://localhost:8000/api/echo-bytes?sr={framerate}&ch={channels}&fmt={fmt}"
    headers = {"Content-Type": "application/octet-stream"}
    
    print(f"\nОтправка на {url}...")
    
    try:
        response = requests.post(url, data=pcm_data, stream=True, headers=headers)
        response.raise_for_status()
        
        # Проверка наличия JSON с распознанным текстом в заголовках или первом чанке
        content_type = response.headers.get('Content-Type', '')
        
        pcm_chunks = []
        chunk_count = 0
        
        print("\nПолучение ответа:")
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                chunk_count += 1
                pcm_chunks.append(chunk)
                print(f"Фрейм {chunk_count}: {len(chunk)} байт, время: {time.time():.3f}")
        
        # Сохранение в WAV
        pcm_output = b''.join(pcm_chunks)
        
        try:
            # Если первые байты - это JSON
            if pcm_output.startswith(b'{'):
                json_end = pcm_output.find(b'\n')
                if json_end > 0:
                    import json
                    text_data = json.loads(pcm_output[:json_end])
                    print(f"\n📝 Распознанный текст: {text_data.get('text', 'N/A')}")
                    if 'segments' in text_data:
                        print(f"Сегменты: {text_data['segments']}")
                    pcm_output = pcm_output[json_end+1:]
        except:
            pass
        
        with wave.open(output_file, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(framerate)
            wav_file.writeframes(pcm_output)
        
        print(f"\n✓ Аудио сохранено в {output_file}")
        print(f"Всего фреймов: {chunk_count}, размер: {len(pcm_output)} байт")
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Ошибка запроса: {e}")

if __name__ == "__main__":
    echo_bytes_http()
