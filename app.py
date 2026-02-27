from fastapi import FastAPI, WebSocket
import json
import asyncio
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sarvamai import AsyncSarvamAI
import io as _io

from chat import stream_chat_with_history, VOICE_AGENT_SYSTEM_PROMPT
from translate import translation_pipeline
from tts import tts_stream_sarvam, tts_stream_elevenlabs

# ── Persistent Sarvam client (reused across all sessions) ──
sarvam_client = AsyncSarvamAI(
    api_subscription_key=os.getenv("SARVAM_API_KEY")
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

# ──────────────────────────────────────────────────────────
# Conversational Voice Agent  /ws/agent
# ──────────────────────────────────────────────────────────

AGENT_SAMPLE_RATE = 16000

@app.websocket("/ws/agent")
async def websocket_agent(websocket: WebSocket):
    await websocket.accept()

    # Per-session conversation history
    conversation_history: list = [
        {"role": "system", "content": VOICE_AGENT_SYSTEM_PROMPT}
    ]

    active_pipeline: asyncio.Task | None = None

    async def ws_send(data: dict):
        """Send JSON, ignoring errors if the socket is already closed."""
        try:
            await websocket.send_json(data)
        except Exception:
            pass

    # ── STT helper ──────────────────────────────────────────
    async def transcribe_wav(wav_bytes: bytes):
        print(f"[STT] calling Sarvam with {len(wav_bytes)} bytes", flush=True)
        resp = await sarvam_client.speech_to_text.translate(
            file=_io.BytesIO(wav_bytes), model="saaras:v3"
        )
        print(f"[STT] Sarvam returned: transcript={repr(resp.transcript)} lang={resp.language_code}", flush=True)
        return resp.transcript, resp.language_code

    # ── Pipeline workers ────────────────────────────────────
    async def _translation_worker(translate_q, tts_q, target_lang):
        try:
            while True:
                item = await translate_q.get()
                if item is None:
                    await tts_q.put(None)
                    break
                original_text, stream_count = item

                # For English, skip translation entirely → forward to TTS immediately
                if not target_lang or target_lang == "en-IN" or target_lang == "":
                    translated = original_text
                else:
                    try:
                        translated = await translation_pipeline(
                            original_text, source_lang="en-IN", target_lang=target_lang
                        )
                    except Exception as e:
                        print(f"Translation error: {e}")
                        translated = original_text

                await ws_send({
                    "type": "bot_text",
                    "text": translated,
                    "stream_count": stream_count,
                })

                await tts_q.put((translated, stream_count))
        except asyncio.CancelledError:
            await tts_q.put(None)
            raise

    async def _tts_worker(tts_q, target_lang):
        """Streaming TTS — ElevenLabs for English, Sarvam for all other languages."""
        lang = target_lang or "en-IN"
        use_elevenlabs = lang in ("en-IN", "en-US", "en-GB", "en")

        try:
            while True:
                item = await tts_q.get()
                if item is None:
                    break
                text, _ = item
                if not text.strip():
                    continue

                try:
                    if use_elevenlabs:
                        try:
                            async for audio_b64 in tts_stream_elevenlabs(text):
                                await ws_send({
                                    "type": "audio_chunk",
                                    "audio": audio_b64,
                                })
                        except Exception as el_err:
                            print(f"ElevenLabs TTS failed, falling back to Sarvam: {el_err}")
                            await ws_send({"type": "tts_warning", "message": "ElevenLabs unavailable, using Sarvam"})
                            async for audio_b64 in tts_stream_sarvam(text, lang="en-IN"):
                                await ws_send({
                                    "type": "audio_chunk",
                                    "audio": audio_b64,
                                })
                    else:
                        async for audio_b64 in tts_stream_sarvam(text, lang=lang):
                            await ws_send({
                                "type": "audio_chunk",
                                "audio": audio_b64,
                            })
                except Exception as e:
                    print(f"TTS error for sentence: {e}")
                    await ws_send({"type": "tts_error", "message": str(e)})

        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"TTS worker error: {e}")

    # ── Full turn pipeline ───────────────────────────────────
    async def run_pipeline(user_text: str, target_lang: str):
        translate_q: asyncio.Queue = asyncio.Queue()
        tts_q: asyncio.Queue = asyncio.Queue()
        response_parts: list[str] = []

        conversation_history.append({"role": "user", "content": user_text})
        print(f"[Pipeline] Starting — text={repr(user_text)} lang={target_lang}", flush=True)

        trans_task = asyncio.create_task(
            _translation_worker(translate_q, tts_q, target_lang)
        )
        tts_task = asyncio.create_task(
            _tts_worker(tts_q, target_lang)
        )

        try:
            flush_count = 0

            async for chunk_json in stream_chat_with_history(conversation_history):
                chunk = json.loads(chunk_json)
                sentence = chunk["response"]

                if sentence.strip():
                    response_parts.append(sentence)
                    flush_count += 1
                    await translate_q.put((sentence, flush_count))

            await translate_q.put(None)          # signal end of stream
            await asyncio.gather(trans_task, tts_task)

            # Commit assistant turn to history
            conversation_history.append({
                "role": "assistant",
                "content": " ".join(response_parts),
            })
            await ws_send({"type": "done"})

        except asyncio.CancelledError:
            trans_task.cancel()
            tts_task.cancel()
            await asyncio.gather(trans_task, tts_task, return_exceptions=True)
            # Keep partial response in history so context isn't lost
            if response_parts:
                conversation_history.append({
                    "role": "assistant",
                    "content": " ".join(response_parts),
                })
            raise
        except Exception as e:
            print(f"Pipeline error: {e}")
            trans_task.cancel()
            tts_task.cancel()
            await asyncio.gather(trans_task, tts_task, return_exceptions=True)

    # ── Main receive loop ────────────────────────────────────
    # Client sends:
    #   text frame  → JSON control message  (speech_start / playback_started / playback_ended)
    #   binary frame → complete WAV segment  (produced by client-side Silero VAD)
    try:
        while True:
            msg = await websocket.receive()

            # ── Control messages ─────────────────────────────
            if "text" in msg and msg["text"]:
                try:
                    ctrl = json.loads(msg["text"])
                except Exception:
                    continue

                if ctrl.get("type") == "speech_start":
                    # VAD confirmed user started speaking — cancel bot immediately
                    if active_pipeline and not active_pipeline.done():
                        active_pipeline.cancel()
                        try:
                            await active_pipeline
                        except (asyncio.CancelledError, Exception):
                            pass
                        await ws_send({"type": "interrupted"})

            # ── Speech segment (WAV bytes from client VAD) ────
            elif "bytes" in msg and msg["bytes"]:
                # Always cancel any still-running pipeline before starting a new one.
                # speech_start should have done this already, but guard against edge
                # cases (first utterance, VAD misfire sending double speech_end, etc.)
                if active_pipeline and not active_pipeline.done():
                    active_pipeline.cancel()
                    try:
                        await active_pipeline
                    except (asyncio.CancelledError, Exception):
                        pass

                wav_bytes = msg["bytes"]
                await ws_send({"type": "processing"})
                try:
                    print(f"[STT] Sending {len(wav_bytes)} bytes to Sarvam...", flush=True)
                    transcript, lang = await asyncio.wait_for(
                        transcribe_wav(wav_bytes), timeout=20.0
                    )
                    lang = lang or "en-IN"
                    print(f"[STT] transcript={repr(transcript)} lang={lang}", flush=True)

                    if transcript.strip():
                        await ws_send({"type": "transcript", "text": transcript})
                        active_pipeline = asyncio.create_task(
                            run_pipeline(transcript, lang)
                        )
                    else:
                        await ws_send({"type": "idle"})

                except asyncio.TimeoutError:
                    print("[STT] Timed out after 20s", flush=True)
                    await ws_send({"type": "error", "message": "STT timed out"})
                except Exception as e:
                    print(f"[STT] Error: {e}")
                    await ws_send({"type": "error", "message": str(e)})

            elif msg.get("type") == "websocket.disconnect":
                break

    except Exception as e:
        print(f"Agent WebSocket closed: {e}")
    finally:
        if active_pipeline and not active_pipeline.done():
            active_pipeline.cancel()
            try:
                await active_pipeline
            except (asyncio.CancelledError, Exception):
                pass