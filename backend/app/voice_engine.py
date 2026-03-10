"""
Kineti-AI Voice Engine
TTS synthesis and echo suppression.
"""

import time
import difflib
import edge_tts

from app.config import VOICE_NAME, ECHO_SIMILARITY_THRESHOLD

# Track when AI last spoke for time-based echo suppression
_last_ai_speak_time = 0.0
_last_ai_text = ""


def record_ai_response(text):
    """Call this after AI generates a response to track echo timing."""
    global _last_ai_speak_time, _last_ai_text
    _last_ai_speak_time = time.time()
    _last_ai_text = text


def is_echo(user_text, last_text):
    """Prevents the AI from hearing itself via the speakers.
    Uses multiple strategies: time-based, similarity, substring matching.
    """
    if not user_text or not last_text:
        return False
    u_clean = user_text.lower().strip()
    l_clean = last_text.lower().strip()

    # Strategy 1: Time-based — if AI spoke within last 3 seconds, very likely echo
    time_since_ai = time.time() - _last_ai_speak_time
    if time_since_ai < 3.0 and _last_ai_text:
        ai_clean = _last_ai_text.lower().strip()
        # Check if any significant fragment of AI text appears in user text
        if len(u_clean) > 5:
            # Check word overlap
            ai_words = set(ai_clean.split())
            user_words = set(u_clean.split())
            overlap = len(ai_words & user_words)
            if overlap >= 3:
                print(f"🔇 Echo Suppressed (time={time_since_ai:.1f}s, word_overlap={overlap})")
                return True

    # Strategy 2: Exact match or long substring
    if len(u_clean) > 10 and u_clean in l_clean:
        print(f"🔇 Echo Suppressed (substring match)")
        return True

    # Strategy 3: Any 8+ word fragment from AI text appears in user text
    if _last_ai_text:
        ai_words_list = _last_ai_text.lower().split()
        for i in range(len(ai_words_list) - 5):
            fragment = " ".join(ai_words_list[i:i+6])
            if fragment in u_clean:
                print(f"🔇 Echo Suppressed (fragment: '{fragment[:30]}...')")
                return True

    # Strategy 4: Fuzzy similarity match (lowered threshold to catch more echoes)
    # But only apply this if the user text is decently long, otherwise short words
    # like "Start" or "Hello" get swallowed by long AI sentences.
    if len(u_clean) > 8:
        ratio = difflib.SequenceMatcher(None, u_clean, l_clean).ratio()
        if ratio > ECHO_SIMILARITY_THRESHOLD:
            print(f"🔇 Echo Suppressed (Text Length: {len(u_clean)}, Similarity: {ratio:.2f})")
            return True

    return False


async def synthesize_speech(text, output_path="response.mp3"):
    """Generate TTS audio file from text using Sarvam AI."""
    record_ai_response(text)  # Track for echo suppression
    import requests
    import base64
    from app.config import SARVAM_API_KEY

    url = "https://api.sarvam.ai/text-to-speech"
    
    payload = {
        "inputs": [text],
        "target_language_code": "hi-IN",
        "speaker": "abhilash",
        "speech_sample_rate": 8000
    }

    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }

    try:
        # Run synchronous requests call in a thread pool to avoid blocking async event loop
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.post(url, json=payload, headers=headers))
        
        if response.status_code == 200:
            data = response.json()
            if "audios" in data and len(data["audios"]) > 0:
                audio_base64 = data["audios"][0]
                audio_bytes = base64.b64decode(audio_base64)
                with open(output_path, "wb") as f:
                    f.write(audio_bytes)
                print("🔊 Sarvam TTS generated successfully")
                return output_path
            else:
                print(f"⚠️ Sarvam API responded but 'audios' key missing: {data}")
        else:
            print(f"⚠️ Sarvam API Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"⚠️ Sarvam request failed: {e}")
        
    return None

