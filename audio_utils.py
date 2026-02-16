import io
from typing import List
from pydub import AudioSegment

def split_audio_bytes(audio_bytes: bytes, chunk_length: int = 30) -> List[bytes]:
    # pydub automatically detects sample_width, frame_rate, etc.
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    duration_ms = len(audio)
    chunk_length_ms = chunk_length * 1000 

    if duration_ms <= chunk_length_ms:
        return [audio_bytes]

    chunks = []
    for start_ms in range(0, duration_ms, chunk_length_ms):
        chunk = audio[start_ms : start_ms + chunk_length_ms]
        buf = io.BytesIO()
        # You can specify format="mp3" here if you want smaller chunks!
        chunk.export(buf, format="wav") 
        chunks.append(buf.getvalue())

    return chunks