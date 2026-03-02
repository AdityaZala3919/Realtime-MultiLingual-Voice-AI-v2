import os
from sarvamai import AsyncSarvamAI
import io as _io

sarvam_client = AsyncSarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

async def transcribe_wav(wav_bytes: bytes):
    resp = await sarvam_client.speech_to_text.translate(
        file=_io.BytesIO(wav_bytes),
        model="saaras:v3"
    )
    return resp.transcript, resp.language_code