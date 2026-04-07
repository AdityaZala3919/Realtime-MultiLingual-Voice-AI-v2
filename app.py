from fastapi import FastAPI, WebSocket
import json
import asyncio
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from chat import stream_chat_with_history, VOICE_AGENT_SYSTEM_PROMPT
from translate import translation_pipeline
from tts import tts_stream_sarvam
from stt import transcribe_wav
from vad import _SessionVAD, CHUNK_SAMPLES, SAMPLE_RATE

STT_RATE_INR_PER_HOUR = 30.0
TTS_RATE_INR_PER_10K_CHARS = 30.0
TRANSLATION_RATE_INR_PER_10K_CHARS = 20.0
LLM_INPUT_COST_PER_MILLION_TOKENS = 37.5
LLM_OUTPUT_COST_PER_MILLION_TOKENS = 150.0

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://realtime-voice-ai.adityazala.dev",
        "https://www.realtime-voice-ai.adityazala.dev",
    ],
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

@app.websocket("/ws/agent")
async def websocket_agent(websocket: WebSocket):
    await websocket.accept()

    # Per-session conversation history
    conversation_history: list = [
        {"role": "system", "content": VOICE_AGENT_SYSTEM_PROMPT}
    ]

    active_pipeline: asyncio.Task | None = None
    session_costs = {
        "speech_to_text": 0.0,
        "translation": 0.0,
        "tts": 0.0,
        "llm": 0.0,
    }
    session_translation_chars = 0
    session_tts_chars = 0
    session_llm_input_tokens = 0
    session_llm_output_tokens = 0

    async def ws_send(data: dict):
        """Send JSON, ignoring errors if the socket is already closed."""
        try:
            await websocket.send_json(data)
        except Exception:
            pass

    def _stt_duration_seconds() -> float:
        return (session_vad.speech_chunk_count * CHUNK_SAMPLES) / SAMPLE_RATE

    def _stt_cost_inr(duration_seconds: float) -> float:
        return round((duration_seconds / 3600.0) * STT_RATE_INR_PER_HOUR, 4)

    def _tts_cost_inr(chars_count: int) -> float:
        return round((chars_count / 10000.0) * TTS_RATE_INR_PER_10K_CHARS, 4)
    
    def _translation_cost_inr(chars_count: int) -> float:
        return round((chars_count / 10000.0) * TRANSLATION_RATE_INR_PER_10K_CHARS, 4)

    def _llm_cost_inr(input_tokens: int, output_tokens: int) -> tuple:
        """Calculate LLM cost for input and output tokens. Returns (total_cost, input_cost, output_cost)."""
        input_cost = round((input_tokens / 1000000.0) * LLM_INPUT_COST_PER_MILLION_TOKENS, 4)
        output_cost = round((output_tokens / 1000000.0) * LLM_OUTPUT_COST_PER_MILLION_TOKENS, 4)
        total_cost = round(input_cost + output_cost, 4)
        return total_cost, input_cost, output_cost

    def _build_cost_update(duration_seconds: float, turn_stt_cost: float) -> dict:
        return {
            "type": "cost_update",
            "components": [
                {
                    "key": "speech_to_text",
                    "label": "Speech to Text",
                    "rate_inr_per_hour": STT_RATE_INR_PER_HOUR,
                    "usage_seconds": round(duration_seconds, 2),
                    "cost_inr": round(turn_stt_cost, 4),
                },
                {
                    "key": "translation",
                    "label": "Translation",
                    "rate_inr_per_10k_chars": TRANSLATION_RATE_INR_PER_10K_CHARS,
                    "usage_chars": session_translation_chars,
                    "cost_inr": round(session_costs["translation"], 4),
                },
                {
                    "key": "tts",
                    "label": "Text to Speech",
                    "rate_inr_per_10k_chars": TTS_RATE_INR_PER_10K_CHARS,
                    "usage_chars": session_tts_chars,
                    "cost_inr": round(session_costs["tts"], 4),
                },
                {
                    "key": "llm",
                    "label": "LLM (Groq)",
                    "rate_inr_input_per_million": LLM_INPUT_COST_PER_MILLION_TOKENS,
                    "rate_inr_output_per_million": LLM_OUTPUT_COST_PER_MILLION_TOKENS,
                    "input_tokens": session_llm_input_tokens,
                    "output_tokens": session_llm_output_tokens,
                    "cost_inr": round(session_costs["llm"], 4),
                },
            ],
            "session_total_inr": round(sum(session_costs.values()), 4),
            "speech_to_text_total_inr": round(session_costs["speech_to_text"], 4),
        }

    # ── Pipeline workers ────────────────────────────────────
    async def _translation_worker(translate_q, tts_q, target_lang):
        nonlocal session_translation_chars
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
                        chars_count = len(original_text)
                        translate_cost = _translation_cost_inr(chars_count)
                        session_translation_chars += chars_count
                        session_costs["translation"] += translate_cost
                        translated = await translation_pipeline(
                            original_text, source_lang="en-IN", target_lang=target_lang
                        )
                        print(
                            f"[Cost] Translation chars={chars_count} rate=₹{TRANSLATION_RATE_INR_PER_10K_CHARS}/10K chars chunk=₹{translate_cost:.4f} session_total=₹{session_costs['translation']:.4f}",
                            flush=True,
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
        nonlocal session_tts_chars
        lang = target_lang or "en-IN"
        try:
            while True:
                item = await tts_q.get()
                if item is None:
                    break
                text, _ = item
                if not text.strip():
                    continue
                chars_count = len(text)
                tts_cost = _tts_cost_inr(chars_count)
                session_tts_chars += chars_count
                session_costs["tts"] += tts_cost
                print(
                    f"[Cost] TTS chars={chars_count} rate=₹{TTS_RATE_INR_PER_10K_CHARS}/10K chars chunk=₹{tts_cost:.4f} session_total=₹{session_costs['tts']:.4f}",
                    flush=True,
                )
                await ws_send(_build_cost_update(0.0, 0.0))
                try:
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
        nonlocal session_llm_input_tokens, session_llm_output_tokens
        
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
                
                # Handle token usage message from LLM
                if chunk.get("type") == "token_usage":
                    input_tokens = chunk.get("input_tokens", 0)
                    output_tokens = chunk.get("output_tokens", 0)
                    session_llm_input_tokens += input_tokens
                    session_llm_output_tokens += output_tokens
                    
                    llm_total_cost, llm_input_cost, llm_output_cost = _llm_cost_inr(input_tokens, output_tokens)
                    session_costs["llm"] += llm_total_cost
                    
                    print(
                        f"[Cost] LLM input_tokens={input_tokens} rate=₹{LLM_INPUT_COST_PER_MILLION_TOKENS}/M tokens input_cost=₹{llm_input_cost:.4f} output_tokens={output_tokens} rate=₹{LLM_OUTPUT_COST_PER_MILLION_TOKENS}/M tokens output_cost=₹{llm_output_cost:.4f} total_cost=₹{llm_total_cost:.4f} session_total=₹{session_costs['llm']:.4f}",
                        flush=True,
                    )
                    await ws_send(_build_cost_update(0.0, 0.0))
                    continue
                
                sentence = chunk.get("response")
                if not sentence:
                    continue

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

    # ── Server-side VAD state ────────────────────────────────
    session_vad = _SessionVAD()

    async def _cancel_pipeline():
        nonlocal active_pipeline
        if active_pipeline and not active_pipeline.done():
            active_pipeline.cancel()
            try:
                await active_pipeline
            except (asyncio.CancelledError, Exception):
                pass

    async def _handle_speech_end():
        """Called by the VAD loop when a complete speech segment is ready."""
        nonlocal active_pipeline
        wav_bytes = session_vad.get_wav_bytes()
        duration_seconds = _stt_duration_seconds()
        turn_stt_cost = _stt_cost_inr(duration_seconds)
        session_costs["speech_to_text"] += turn_stt_cost
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

            await ws_send(_build_cost_update(duration_seconds, turn_stt_cost))
            print(
                f"[Cost] STT duration={duration_seconds:.2f}s rate=₹{STT_RATE_INR_PER_HOUR}/hr turn=₹{turn_stt_cost:.4f} session_total=₹{session_costs['speech_to_text']:.4f}",
                flush=True,
            )

        except asyncio.TimeoutError:
            print("[STT] Timed out after 20s", flush=True)
            await ws_send({"type": "error", "message": "STT timed out"})
        except Exception as e:
            print(f"[STT] Error: {e}")
            await ws_send({"type": "error", "message": str(e)})

    # ── Main receive loop ────────────────────────────────────
    # Client sends:
    #   binary frame → raw float32 PCM chunk (512 samples @ 16 kHz = 2048 bytes)
    #                  streamed continuously while the session is active.
    #   text frame   → JSON control message (playback_started / playback_ended)
    #
    # Server sends:
    #   speech_start  – VAD detected onset of user speech (for client UI)
    #   processing    – speech segment complete, STT running
    #   transcript    – STT result
    #   bot_text      – LLM response chunk
    #   audio_chunk   – TTS audio
    #   done          – full response delivered
    #   interrupted   – response cancelled due to new user speech
    #   idle          – blank transcript after VAD fired
    #   error         – server-side error description
    try:
        while True:
            msg = await websocket.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            # ── Control messages (playback tracking etc.) ──────
            if "text" in msg and msg["text"]:
                # Currently no inbound control messages are needed from the
                # client in server-VAD mode, but we keep parsing for
                # forward compatibility.
                try:
                    json.loads(msg["text"])  # validate JSON, ignore content
                except Exception:
                    pass
                continue

            # ── Raw float32 PCM chunk from client microphone ───
            if "bytes" not in msg or not msg["bytes"]:
                continue

            chunk_bytes: bytes = msg["bytes"]

            # Silero expects exactly CHUNK_SAMPLES float32 values.
            # Silently skip malformed chunks.
            if len(chunk_bytes) != CHUNK_SAMPLES * 4:
                continue

            # Run VAD — this is CPU-bound but fast (~0.5 ms), safe on event loop
            event = await asyncio.get_event_loop().run_in_executor(
                None, session_vad.process, chunk_bytes
            )

            if event == "start":
                print("[VAD] Speech start", flush=True)
                # Interrupt any active bot response
                pipeline_was_active = (
                    active_pipeline and not active_pipeline.done()
                )
                await _cancel_pipeline()
                await ws_send({"type": "speech_start"})
                if pipeline_was_active:
                    await ws_send({"type": "interrupted"})

            elif event == "end":
                print("[VAD] Speech end", flush=True)
                await _handle_speech_end()

            elif event == "misfire":
                print("[VAD] Misfire (too short)", flush=True)
                await ws_send({"type": "idle"})

    except Exception as e:
        print(f"Agent WebSocket closed: {e}")
    finally:
        await _cancel_pipeline()