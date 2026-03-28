"""
Minimal ElevenLabs TTS: text → speaker output

"""

import os
import sys
import time
from itertools import chain

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import stream

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing ELEVENLABS_API_KEY in .env")

client = ElevenLabs(api_key=API_KEY)

VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # default voice
MODEL_ID = "eleven_flash_v2_5"     # low-latency model


# -----------------------------------------------------------------------------
# Core function
# -----------------------------------------------------------------------------

def speak(text: str, *, log_latency: bool = False) -> None:
    if not text or not text.strip():
        print("No text provided.")
        return

    audio_stream = client.text_to_speech.stream(
        text=text.strip(),
        voice_id=VOICE_ID,
        model_id=MODEL_ID,
        output_format="mp3_44100_128",
    )

    if not log_latency:
        stream(audio_stream)
        return

    t0 = time.perf_counter()
    try:
        first = next(audio_stream)
    except StopIteration:
        print("No audio chunks received from the API.")
        return
    t_first = time.perf_counter()

    stream(chain([first], audio_stream))
    t_end = time.perf_counter()

    print(
        f"Latency - first audio chunk: {(t_first - t0) * 1000:.1f} ms "
        f"(time until first bytes; playback can start here)"
    )
    print(
        f"Latency - total until playback finished: {(t_end - t0) * 1000:.1f} ms "
        f"(includes full audio duration, not just the API)"
    )


# -----------------------------------------------------------------------------
# Main (for testing)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    log_latency = "--latency" in sys.argv
    phrase = input("Enter text to speak: ")

    try:
        speak(phrase, log_latency=log_latency)
        print("Process complete")
    except Exception as e:
        print(f"Error: {e}")