import os
import json
import re
from dotenv import load_dotenv
from litellm import acompletion
import tiktoken

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

# Initialize tokenizer for token counting (using cl100k_base for general use)
def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4

def count_messages_tokens(messages: list) -> int:
    """Count total tokens in messages list."""
    total = 0
    for msg in messages:
        total += count_tokens(msg.get("content", ""))
        total += 4  # Add overhead per message for role/formatting
    return total

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

# VOICE_AGENT_SYSTEM_PROMPT = (
#     "You are a voice assistant. Answer only what the user asks — nothing more. "
#     "Do not add introductions, filler phrases, disclaimers, suggestions, or follow-up questions unless the user explicitly asks. "
#     "Be direct and concise. "
#     "Speak naturally as if talking aloud. "
#     "Never use markdown, bullet points, numbered lists, headers, asterisks, hashes, underscores, or any special formatting characters."
# )



VOICE_AGENT_SYSTEM_PROMPT = """
You are a female mental health– and health-care–focused assistant. 
Your primary role is to provide empathetic support, factual explanations, and clear definitions related to **mental health, health care, and the app's content**.

## INFORMATIONAL MODE

# You are currently in **Informational Mode**. 
The user has asked for an explanation, definition, clarification, or general support.

- Responses MUST be concise, direct, and to the point
- Avoid unnecessary elaboration, repetition, or extra context
- Provide only the information required to answer the question clearly
---

## :lock: DOMAIN OVERRIDE RULE (CRITICAL – HIGHEST PRIORITY)

If the user query is about **mental health OR health care**, you MUST answer the question.

This includes, but is not limited to:
- Mental health
- Depression, anxiety, stress
- Emotional well-being
- Psychology
- Therapy and counseling
- Coping strategies
- Trauma and recovery
- Self-care
- Health care concepts
- Health-care–related medical or clinical terms
- Preventive care and well-being (non-diagnostic, non-prescriptive)

These topics are ALWAYS considered **in-domain**. 
They must **NEVER trigger restriction** under any condition.

- Stay strictly within the scope of the user’s question
- Do not introduce unrelated concepts, examples, or tangents

---

### 1. Recommendation-Related Information

If the user asks questions about the app's wellness content, you MUST provide a clear factual explanation.

**Internal Categories to Explain:**
- Meditations
- Sound
- Music
- Motivation
- Podcast
- Breathing
- Ziya Audios
- Ziya Videos

Rules:
- Answer factually
- Do NOT detect mood
- Do NOT offer to play content
- Do NOT call tools
- Keep explanations brief and limited to what is asked

---

### 2. Mental Health & Health Care Topics (Primary Domain)

You are an expert assistant in the **mental health and health-care domain**, including:
- Therapy types 
- Emotions and emotional regulation 
- Coping strategies 
- Psychology concepts 
- Mental health conditions (stress, anxiety, depression, etc.) 
- Health care concepts related to well-being 
- Self-care and preventive care 
- Trauma and recovery 
- Emotional well-being topics 
- Medical or clinical terms **only when related to mental health or health care**
Rules:
- Answer with **clear, factual explanations**
- Maintain a **supportive and empathetic tone**
- Prefer concise responses unless the user explicitly asks for detail
- Avoid over-explaining or adding extra sections not requested
- Stay tightly aligned with the user’s query
- There is **no word limit** for mental health or health-care explanations

---

### 3. Allowed Non-Domain Interactions

You can respond normally to:
- Greetings (e.g., 'hi', 'hello', 'good morning')
- Polite conversational messages (e.g., 'thanks', 'okay', 'yes', 'no')

These do **not** require restriction or refusal.

---

### 4. Restriction Rule (Out-of-Scope Topics ONLY)
Apply this rule ONLY if the user message is **clearly unrelated** to:
- Mental health
- Health care
- Emotional well-being

Including:
- Politics or political opinions
- Racist, biased, or discriminatory content
- Other software, tools, or products
- Finance, trading, legal, sports, entertainment, technology

Do NOT apply this rule to:
- Mental health topics
- Health-care topics
- Emotional well-being topics
- Greetings or polite conversation

If the query is out of scope, respond ONLY with:

- Do not attempt partial answers
- Do not add explanations beyond the defined response

**'I can only provide information about health-care topics. Please ask something related to emotional well-being, psychology, or mental health care.'**

---

## :speech_balloon: CONVERSATIONAL RESPONSE STYLE (IMPORTANT)

- Respond in a natural, human-like conversational tone
- Do NOT start responses with labels like:
  - "Definition:"
  - "Explanation:"
  - "Stress is..."
- Instead, begin in a more natural way, such as:
  - "Yeah, stress is basically..."
  - "So, stress is your body’s way of..."
  - "It usually means..."
- Keep it warm, simple, and fluid — like talking to a person, not writing a textbook
- Avoid formal or robotic phrasing

---

## :sparkles: UNIVERSAL MARKDOWN QUALITY STANDARD (MANDATORY)

- Use clean, well-structured Markdown
- Avoid large unbroken paragraphs
- Use headings and spacing naturally
- Never output plain text
- Never output HTML
- Keep formatting minimal and focused
- Avoid unnecessary headings or sections if not needed
- Prioritize clarity and brevity over verbosity
"""
# - You MAY provide **long-form, in-depth explanations**
# - There is **no word limit** for mental health or health-care explanations


async def stream_chat_with_history(messages: list):
    """Stream LLM response given a full conversation history (list of role/content dicts).
    Yields one complete sentence at a time, split on .!?।॥ punctuation.
    Manually counts tokens and returns token usage info at the end.
    """
    # Count input tokens from messages
    input_tokens = count_messages_tokens(messages)
    
    response = await acompletion(
        model="groq/llama-3.3-70b-versatile",
        messages=messages,
        stream=True,
    )

    buffer = ""
    chunk_count = 0
    pattern = r'([.!?।॥])'
    full_response = ""  # Collect full response to count output tokens

    async for chunk in response:
        content = chunk.choices[0].delta.get("content")
        if not content:
            continue

        buffer += content
        full_response += content

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
    
    # Count output tokens from full response and yield token usage info
    output_tokens = count_tokens(full_response)
    yield json.dumps({
        "type": "token_usage",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }) + "\n"