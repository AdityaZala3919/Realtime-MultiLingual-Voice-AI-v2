import os
import json
import re
from dotenv import load_dotenv
from litellm import acompletion

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

def strip_markdown(text: str) -> str:
            """
            Strip markdown formatting from text for TTS.
            Removes headers (###), bold/italic (**), bullet points (-), etc.
            Preserves numbers so they can be spoken.
            """
            if text is None:
                text = ""
            # Remove markdown headers (### Header -> Header)
            # But preserve any numbers that follow the header markers
            text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
            # Remove bold/italic markers (**text** or *text* or __text__ or _text_)
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
            text = re.sub(r'\*([^*]+)\*', r'\1', text)
            text = re.sub(r'__([^_]+)__', r'\1', text)
            text = re.sub(r'_([^_]+)_', r'\1', text)
            # Remove bullet points at the start of lines (- item or * item)
            text = re.sub(r'^[\-\*]\s+', '', text, flags=re.MULTILINE)
            # Keep numbered list markers but make them speakable (1. item -> 1, item)
            # This preserves the number for TTS while removing the period that follows it
            # Handle both Arabic numerals (0-9) and Hindi Devanagari numerals (०-९)
            text = re.sub(r'^([\d०-९]+)\.\s+', r'\1, ', text, flags=re.MULTILINE)
            # Remove code blocks and inline code
            text = re.sub(r'```[^`]*```', '', text)
            text = re.sub(r'`([^`]+)`', r'\1', text)
            # Remove links [text](url) -> text
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
            # Replace newlines with spaces for natural speech flow
            text = text.replace('\n', ' ').replace('\r', ' ')
            # Collapse multiple spaces
            text = re.sub(r'\s+', ' ', text).strip()
            return text

VOICE_AGENT_SYSTEM_PROMPT = (
    "You are a voice assistant. Answer only what the user asks — nothing more. "
    "Do not add introductions, filler phrases, disclaimers, suggestions, or follow-up questions unless the user explicitly asks. "
    "Be direct and concise. "
    "Speak naturally as if talking aloud. "
    "Never use markdown, bullet points, numbered lists, headers, asterisks, hashes, underscores, or any special formatting characters."
)

async def stream_chat_with_history(messages: list):
    """Stream LLM response given a full conversation history (list of role/content dicts).
    Yields one complete sentence at a time, split on .!?।॥ punctuation.
    """
    response = await acompletion(
        model="groq/llama-3.3-70b-versatile",
        messages=messages,
        stream=True,
    )

    buffer = ""
    chunk_count = 0
    pattern = r'([.!?।॥])'

    async for chunk in response:
        content = chunk.choices[0].delta.get("content")
        if not content:
            continue

        buffer += content

        # sentence-boundary streaming
        parts = re.split(pattern, buffer)
        # parts looks like: [pre, punct, pre, punct, ..., remainder]
        for i in range(0, len(parts) - 1, 2):
            sentence = (parts[i] + parts[i + 1]).strip()
            if sentence:
                chunk_count += 1
                print(sentence)
                yield json.dumps({
                    "response": strip_markdown(sentence),
                    "stream_count": chunk_count,
                }) + "\n"
        buffer = parts[-1]  # keep the incomplete tail

    # Flush any remaining text that had no sentence-ending punctuation
    if buffer.strip() and re.search(r'[^\d\s\W]', buffer, re.UNICODE):
        chunk_count += 1
        yield json.dumps({
            "response": strip_markdown(buffer.strip()),
            "stream_count": chunk_count,
        }) + "\n"