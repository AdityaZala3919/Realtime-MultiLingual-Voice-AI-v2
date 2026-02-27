import os
import asyncio
import base64
from typing import Any, AsyncIterator, Dict, Optional
import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Sarvam ─────────────────────────────────────────────────────────────────
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STREAM_URL = "https://api.sarvam.ai/text-to-speech/stream"

# ── ElevenLabs ─────────────────────────────────────────────────────────────
ELEVENLABS_API_KEY: str | None = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "CwhRBWXzGAHq8TQ4Fs17")
MODEL_ID: str = os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2")
DEFAULT_STREAM_URL_TEMPLATE: str = (
    "https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream"
)


async def tts_stream_sarvam(
    text,
    lang="en-IN",
    model="bulbul:v3",
    speaker="shubh",
    pace=1.0,
    api_key=SARVAM_API_KEY,
):
    """
    Stream TTS via Sarvam REST streaming endpoint.
    Yields base64-encoded MP3 chunks as they arrive.
    """
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "target_language_code": lang,
        "speaker": speaker,
        "model": model,
        "pace": pace,
        "speech_sample_rate": 22050,
        "output_audio_codec": "mp3",
        "enable_preprocessing": True,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST", SARVAM_STREAM_URL, headers=headers, json=payload
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=4096):
                if chunk:
                    print("="*10, "Sarvam", "="*10)
                    yield base64.b64encode(chunk).decode()
                    # await asyncio.sleep(0.01)  # 10ms between HTTP chunks


async def tts_stream_elevenlabs(
    text: str,
    *,
    api_key: str = ELEVENLABS_API_KEY,
    voice: str = VOICE_ID,
    model: str = MODEL_ID,
    stream_url: Optional[str] = None,
    api_key_header: str = "xi-api-key",
    chunk_size: int = 4096,
    timeout: float = 60.0,
    request_kwargs: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[str]:
    """
    Streams ElevenLabs TTS audio and yields base64-encoded audio chunks.
    """
    if stream_url is None:
        stream_url = DEFAULT_STREAM_URL_TEMPLATE.format(voice=voice)

    headers = {
        api_key_header: api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    payload = {
        "text": text,
        "model_id": model,
    }

    request_kwargs = request_kwargs or {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        async with client.stream(
            "POST",
            stream_url,
            headers=headers,
            json=payload,
            **request_kwargs,
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                if chunk:
                    print("="*10, "11labs", "="*10)
                    yield base64.b64encode(chunk).decode()


async def tts(
    text,
    lang="en-IN",
    model="bulbul:v3",
    speaker="shubh",
    api_key=SARVAM_API_KEY,
):
    """Non-streaming TTS via REST — returns a single base64 MP3 string."""
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "target_language_code": lang,
        "speaker": speaker,
        "model": model,
        "speech_sample_rate": 22050,
        "output_audio_codec": "mp3",
        "enable_preprocessing": True,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(SARVAM_STREAM_URL, headers=headers, json=payload)
        response.raise_for_status()
        return base64.b64encode(response.content).decode()
