import torch
import numpy as np
import struct
from collections import deque
from silero_vad import load_silero_vad, VADIterator

# ── Silero VAD model (loaded once, shared across sessions — stateless) ──
print("[VAD] Loading Silero VAD model...", flush=True)
_vad_model = load_silero_vad()
print("[VAD] Model loaded.", flush=True)

SAMPLE_RATE       = 16000
CHUNK_SAMPLES     = 512          # silero expects 512 samples @ 16 kHz per call (32 ms)
# 800 ms of pre-speech padding  →  800 / 32 = 25 chunks
PRE_SPEECH_CHUNKS = 25
# 150 ms minimum speech to avoid misfires  →  ceil(150 / 32) = 5 chunks
MIN_SPEECH_CHUNKS = 5

def _encode_wav(float32_pcm: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Convert a float32 numpy array to a 16-bit mono WAV byte string."""
    pcm_int16 = np.clip(float32_pcm, -1.0, 1.0)
    pcm_int16 = (pcm_int16 * 32767).astype(np.int16)
    num_bytes = pcm_int16.nbytes
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + num_bytes, b"WAVE",
        b"fmt ", 16, 1, 1,           # PCM, mono
        sample_rate, sample_rate * 2, 2, 16,
        b"data", num_bytes,
    )
    return header + pcm_int16.tobytes()

class _SessionVAD:
    """Per-WebSocket VAD state machine using Silero VADIterator."""

    def __init__(self):
        self.iterator = VADIterator(
            _vad_model,
            threshold=0.7,               # matches frontend positiveSpeechThreshold
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=1400, # matches frontend redemptionMs
            speech_pad_ms=200,
        )
        self.speaking       = False
        self.speech_chunks: list[bytes] = []
        self.speech_chunk_count         = 0
        # Rolling buffer of the last PRE_SPEECH_CHUNKS raw float32 chunks
        self.pre_buffer: deque = deque(maxlen=PRE_SPEECH_CHUNKS)

    def process(self, chunk_bytes: bytes) -> str | None:
        """
        Feed one 512-sample float32 chunk.
        Returns:
          'start'   – speech onset detected (send speech_start to client)
          'end'     – speech offset detected (enough audio accumulated)
          'misfire' – speech too short, discard (send idle to client)
          None      – no state change
        """
        audio = torch.from_numpy(
            np.frombuffer(chunk_bytes, dtype=np.float32).copy()
        )

        if not self.speaking:
            self.pre_buffer.append(chunk_bytes)   # keep rolling window

        result = self.iterator(audio)

        if not self.speaking:
            if result is not None and "start" in result:
                self.speaking         = True
                # Seed speech_chunks with pre-speech padding (includes current chunk)
                self.speech_chunks    = list(self.pre_buffer)
                self.speech_chunk_count = len(self.speech_chunks)
                return "start"
        else:
            self.speech_chunks.append(chunk_bytes)
            self.speech_chunk_count += 1
            if result is not None and "end" in result:
                self.speaking = False
                if self.speech_chunk_count < MIN_SPEECH_CHUNKS:
                    self.speech_chunks      = []
                    self.speech_chunk_count = 0
                    return "misfire"
                return "end"

        return None

    def get_wav_bytes(self) -> bytes:
        """Concatenate accumulated speech chunks and encode as WAV."""
        pcm = np.concatenate([
            np.frombuffer(c, dtype=np.float32) for c in self.speech_chunks
        ])
        return _encode_wav(pcm, SAMPLE_RATE)

    def reset(self):
        self.iterator.reset_states()
        self.speaking           = False
        self.speech_chunks      = []
        self.speech_chunk_count = 0
        self.pre_buffer.clear()