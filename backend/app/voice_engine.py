"""
Kineti-AI Voice Engine
TTS synthesis (Sarvam AI primary, Edge-TTS fallback) and echo suppression.
"""

import time
import difflib
import base64
import requests
import edge_tts

from app.config import VOICE_NAME, ECHO_SIMILARITY_THRESHOLD, SARVAM_API_KEY

# Track when AI last spoke for time-based echo suppression
_last_ai_speak_time = 0.0
_last_ai_text = ""

# Sarvam TTS configuration
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_SPEAKER = "manisha"         # Sweet female voice for Dr. Kineti
SARVAM_MODEL = "bulbul:v2"         # Using v2 since API access supports these speakers
SARVAM_DEFAULT_LANG = "en-IN"      # Default language


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


async def _sarvam_tts(text, output_path, language="en-IN"):
    """Call Sarvam AI TTS API. Returns True on success, False on failure."""
    if not SARVAM_API_KEY:
        print("⚠️ Sarvam API key not set, skipping Sarvam TTS")
        return False
    
    try:
        # Sarvam has a 2500 char limit for bulbul:v3
        if len(text) > 2500:
            text = text[:2500]
        
        payload = {
            "inputs": [text],
            "target_language_code": language,
            "speaker": SARVAM_SPEAKER,
            "model": SARVAM_MODEL,
            "pace": 1.0,
            "enable_preprocessing": True
        }
        headers = {
            "Content-Type": "application/json",
            "api-subscription-key": SARVAM_API_KEY
        }
        
        response = requests.post(SARVAM_TTS_URL, json=payload, headers=headers, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            audios = result.get("audios", [])
            if audios and audios[0]:
                # Sarvam returns base64-encoded WAV audio — decode and save
                audio_bytes = base64.b64decode(audios[0])
                with open(output_path, "wb") as f:
                    f.write(audio_bytes)
                print(f"🔊 Sarvam TTS generated successfully ({len(audio_bytes)} bytes)")
                return True
            else:
                print("⚠️ Sarvam TTS returned empty audio")
                return False
        else:
            print(f"⚠️ Sarvam TTS API error {response.status_code}: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"⚠️ Sarvam TTS request failed: {e}")
        return False


async def synthesize_speech(text, output_path="response.mp3", language="en-IN"):
    """Generate TTS audio file from text. 
    Primary: Sarvam AI (better quality Indian English voice)
    Fallback: Edge-TTS (free, always works)
    Output format stays the same regardless of which engine is used.
    """
    record_ai_response(text)  # Track for echo suppression
    
    # Try Sarvam AI first (better voice quality)
    if await _sarvam_tts(text, output_path, language):
        return True
    
    # Fallback to Edge-TTS if Sarvam fails
    print("🔄 Falling back to Edge-TTS...")
    try:
        communicate = edge_tts.Communicate(text, VOICE_NAME)
        await communicate.save(output_path)
        print("🔊 Edge-TTS generated successfully (fallback)")
        return True
    except Exception as e:
        print(f"⚠️ Edge-TTS fallback also failed: {e}")
        return False


