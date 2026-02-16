from fastapi import FastAPI, Form, WebSocket
from fastapi.responses import StreamingResponse
import json
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from chat import stream_chat, chat
from translate import translation_pipeline
from tts import tts
from audio_utils import split_audio_bytes
from stt import stt

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatResponse(BaseModel):
    message: str
    total_tokens: int
    total_time: float

@app.post("/chat")
async def chat_endpoint(input: str = Form(), stream: bool = Form()):
    if stream is True:
        return StreamingResponse(
            stream_chat(input, stream),
            media_type="text/event-stream"
        )
    response = await chat(input, stream)
    return ChatResponse(
        message=response[0],
        total_tokens=response[1],
        total_time=response[2]
    )
    
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            audio_chunks = split_audio_bytes(audio_bytes)
            input_text, target_lang = await stt(audio_files=audio_chunks)

            async for chunk_json in stream_chat(input_text, stream=True):
                chunk = json.loads(chunk_json)
                original_response = chunk["response"]

                # Send original response immediately
                await websocket.send_json({
                    "type": "original",
                    "text": original_response,
                    "stream_count": chunk["stream_count"]
                })

                # Translate the response chunk
                try:
                    translated = await translation_pipeline(original_response, source_lang="en-IN", target_lang=target_lang)
                except Exception as e:
                    # If translation fails, use original text
                    translated = original_response
                    print(f"Translation error: {e}")

                # Send translated response as soon as it's ready
                await websocket.send_json({
                    "type": "translated",
                    "text": translated,
                    "stream_count": chunk["stream_count"]
                })
                
                audio_chunk_b64 = await tts(text=translated, lang=target_lang)
                # TTS streaming for each translated chunk
                # async for audio_chunk_b64 in tts_stream(translated, lang=target_lang):
                await websocket.send_json({
                    "type": "audio_chunk",
                    "audio": audio_chunk_b64,
                    "stream_count": chunk["stream_count"]
                })
            await websocket.send_json({"done": True})
    except Exception as e:
        print(f"Websocket closed: {e}")
        