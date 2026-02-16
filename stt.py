import os
from sarvamai import AsyncSarvamAI
from typing import List
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

STT_API_KEY = os.getenv("SARVAM_API_KEY")

async def stt(audio_files: List[bytes]):
    client = AsyncSarvamAI(api_subscription_key=STT_API_KEY)
    transcripts = []
    language_codes = []
    
    for audio_bytes in audio_files:
        from io import BytesIO
        audio_stream = BytesIO(audio_bytes)
        response = await client.speech_to_text.translate(
            file=audio_stream,
            model="saaras:v3"
        )
        transcripts.append(response.transcript)
        language_codes.append(response.language_code)
    
    # Concatenate all transcripts
    full_transcript = " ".join(transcripts)
    # Find the majority language_code
    majority_language_code = Counter(language_codes).most_common(1)[0][0] if language_codes else None

    return full_transcript, majority_language_code