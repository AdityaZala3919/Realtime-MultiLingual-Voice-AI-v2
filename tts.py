import os
from sarvamai import AsyncSarvamAI, AudioOutput
from dotenv import load_dotenv

load_dotenv()

TTS_API_KEY = os.getenv("SARVAM_API_KEY")

async def tts_stream(text, lang="en-IN", model="bulbul:v3", speaker="shubh", api_key=TTS_API_KEY):
    client = AsyncSarvamAI(api_subscription_key=api_key)
    async with client.text_to_speech_streaming.connect(model=model) as ws:
        await ws.configure(target_language_code=lang, speaker=speaker)
        await ws.convert(text)
        await ws.flush()
        async for message in ws:
            if isinstance(message, AudioOutput):
                # Yield raw audio chunk (base64-encoded)
                yield message.data.audio
                
async def tts(text, lang="en-IN", model="bulbul:v3", speaker="shubh", api_key=TTS_API_KEY):
    client = AsyncSarvamAI(api_subscription_key=api_key)
    response = await client.text_to_speech.convert(
        target_language_code=lang,
        text=text,
        model=model,
        speaker=speaker
    )
    return response.audios[0]