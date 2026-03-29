from __future__ import annotations
import os, time, io
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()

MODEL = "gemini-2.5-flash"
DEFAULT_PROMPT = (
    "You are assisting a blind or low-vision user. "
    "Only look at the bottom 60 percent of the image. "
    "Simply say if there is a hazard to watch out for and say where it is. "
    "If there is no discernable hazard, just return 'Clear'. "
    "Don't spend too much time trying to describe the environment but rather describe the hazards. "
    "Prioritize people, cars, stairs, doors, poles, crosswalk signals, and nearby obstacles. "
    "If nothing important is present, reply exactly: CLEAR."
)

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
_DEFAULT_TIMEOUT_MS = 120_000
_MAX_RETRIES        = 3
_RETRY_DELAY_SEC    = 1.0

def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")
    return genai.Client(api_key=api_key.strip(), http_options=types.HttpOptions(timeout=_DEFAULT_TIMEOUT_MS))

def analyze_image(image_path, prompt=DEFAULT_PROMPT, model=MODEL, max_retries=_MAX_RETRIES):
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    client = _get_client()
    for attempt in range(max(1, max_retries)):
        try:
            with Image.open(path) as im:
                image = im.convert("RGB")
            response = client.models.generate_content(model=model, contents=[prompt, image])
            text = (getattr(response, "text", None) or "").strip()
            if not text:
                raise ValueError("Empty response from model")
            return text
        except (FileNotFoundError, RuntimeError):
            raise
        except Exception as exc:
            err_str = str(exc)
            # Never retry quota/rate-limit errors — retrying immediately makes it worse
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                raise RuntimeError(f"Gemini quota exceeded: {exc}") from exc
            if attempt < max_retries - 1:
                time.sleep(_RETRY_DELAY_SEC)
            else:
                raise RuntimeError(f"Gemini failed after {max_retries} attempts: {exc}") from exc

class TTSEngine:
    def __init__(self):
        self._el_client  = None
        self._chars_used = 0
        if ELEVENLABS_API_KEY:
            try:
                from elevenlabs.client import ElevenLabs
                self._el_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
                print("TTS: ElevenLabs active")
            except ImportError:
                print("TTS: elevenlabs not installed")
        else:
            print("TTS: no ELEVENLABS_API_KEY in .env")

    def speak(self, text):
        if not self._el_client:
            print(f"  [no TTS engine] {text}")
            return
        try:
            import pygame
            audio_stream = self._el_client.text_to_speech.convert(
                text=text,
                voice_id="JBFqnCBsd6RMkjVDRZzb",
                model_id="eleven_turbo_v2",
            )
            audio_bytes = b"".join(audio_stream)
            pygame.mixer.init()
            pygame.mixer.music.load(io.BytesIO(audio_bytes))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            self._chars_used += len(text)
        except Exception as e:
            print(f"  ElevenLabs error: {e}")

def run(image_path, prompt=DEFAULT_PROMPT):
    tts = TTSEngine()
    print(f"model  : {MODEL}")
    print(f"image  : {image_path}")
    print("---")
    t0 = time.perf_counter()
    result = analyze_image(image_path, prompt)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"latency: {elapsed_ms:.1f}ms")
    print(f"result : {result}")
    print("---")
    tts.speak(result)
    return result

def main():
    import sys
    image_arg = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"
    run(image_arg)

if __name__ == "__main__":
    main()
