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

async def chat(input_text: str, stream: bool):   
    response = await acompletion(
        model="groq/llama-3.1-8b-instant",
        messages=[{"content": input_text, "role": "user"}],
        stream=stream
    )
    clean_text = strip_markdown(response.choices[0].message.content)
    return clean_text, response.usage.total_tokens, response.usage.total_time

async def stream_chat(input_text: str, stream: bool):
    response = await acompletion(
        model="groq/llama-3.3-70b-versatile",
        messages=[{"content": input_text, "role": "user"}],
        stream=stream
    )
    
    buffer, chunk_count, token_count = "", 0, 0
    pattern = r'([.!?。！？])'
    TOKEN_THRESHOLD = 50

    async for chunk in response:
        content = chunk.choices[0].delta.get("content")
        if content is None:
            content = ""
        buffer += content

        token_count += len(content.split())
        parts = re.split(pattern, buffer)
        chunk_ready = False

        for i in range(0, len(parts) - 1, 2):
            sentence = (parts[i] + parts[i+1]).strip()
            if sentence:
                chunk_count += 1
                yield json.dumps({"response": strip_markdown(sentence), "stream_count": chunk_count}) + "\n"
                token_count = 0
                chunk_ready = True
        buffer = parts[-1]

        if not chunk_ready and token_count >= TOKEN_THRESHOLD and buffer.strip():
            chunk_count += 1
            yield json.dumps({"response": strip_markdown(buffer.strip()), "stream_count": chunk_count}) + "\n"
            buffer = ""
            token_count = 0

    if buffer.strip():
        chunk_count += 1
        yield json.dumps({"response": strip_markdown(buffer.strip()), "stream_count": chunk_count}) + "\n"
        # yield chunk.model_dump_json() + "\n"