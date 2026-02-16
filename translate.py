import os
from sarvamai import AsyncSarvamAI
from dotenv import load_dotenv

load_dotenv()

TRANSLATE_API_KEY = os.getenv("SARVAM_API_KEY")

async def translation_pipeline(text: str, source_lang: str = "en-IN", target_lang: str = "gu-IN"):
    client = AsyncSarvamAI(api_subscription_key=TRANSLATE_API_KEY)
    response = await client.text.translate(
        input=text,
        source_language_code=source_lang,
        target_language_code=target_lang,
        speaker_gender="Male",
        model="sarvam-translate:v1"# "mayura:v1"
    )
    return response.translated_text