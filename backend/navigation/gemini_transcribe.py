"""
Gemini-only transcription of user WAV (voice commands). No vision.
"""

from __future__ import annotations

from google.genai import types

from backend.vlm_with_audio import MODEL, _get_client

_TRANSCRIBE_PROMPT = (
    "Transcribe the spoken words in this audio exactly, in English. "
    "Reply with ONLY the transcript text. No quotes, labels, or extra commentary."
)


def transcribe_wav(wav_bytes: bytes) -> str:
    if not wav_bytes or len(wav_bytes) < 200:
        return ""
    audio_part = types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")
    response = _get_client().models.generate_content(
        model=MODEL,
        contents=[_TRANSCRIBE_PROMPT, audio_part],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return (getattr(response, "text", None) or "").strip()
