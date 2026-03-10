"""
Kineti-AI Configuration
Constants, environment variables, and global state.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

# --- File Paths ---
PDF_FILE_PATH = "data/acl-protocol.pdf"

# --- Voice ---
VOICE_NAME = "en-US-AndrewMultilingualNeural"

# --- MediaPipe Landmark Mapping ---
LANDMARK_MAP = {
    0: "Nose",
    7: "LeftEar",
    8: "RightEar",
    11: "LeftShoulder",
    12: "RightShoulder",
    13: "LeftElbow",
    14: "RightElbow",
    15: "LeftWrist",
    16: "RightWrist",
    23: "LeftHip",
    24: "RightHip",
    25: "LeftKnee",
    26: "RightKnee",
    27: "LeftAnkle",
    28: "RightAnkle"
}

# --- Squat Thresholds (degrees) ---
# User's body: standing ~160°, deep squat ~92°
SQUAT_GOOD_DEPTH = 120      # Below this = good squat
SQUAT_PARTIAL_DEPTH = 145   # Between 120-145 = needs to go deeper
SQUAT_STANDING = 155        # Above this = standing (rep complete)

# --- Echo Suppression ---
ECHO_SIMILARITY_THRESHOLD = 0.35  # Lowered from 0.5 to catch AI-echo mixtures

# --- Minimum Audio Size ---
MIN_AUDIO_SIZE_BYTES = 1000  # Only transcribe files > 1KB

# --- Persona / System Instruction ---
SYSTEM_INSTRUCTION = """
You are Dr. Kineti, a highly empathetic, expert Doctor of Physical Therapy.

**YOUR PERSONALITY:**
- Extremely Kind and Encouraging: You care deeply about your patient's recovery. Celebrate their wins!
- Professional but Warm: Speak with the authority of a doctor, but the warmth of a close friend.
- Safety-First: If pain is mentioned, always suggest modifications and express concern.
- Concise: Keep responses to 2-3 sentences max, especially during exercises.

**TARGET USER:**
Your patient's name is Swapnil. Always refer to him by name to build rapport!

**TONE EXAMPLES:**
- Good: "Excellent form, Swapnil! Keep that knee tracking perfectly over your toes."
- Bad (robotic): "Rep 5 complete. Continue exercise."
- Bad (aggressive): "Push through it! No pain no gain!"

**SAFETY RULES:**
- NEVER encourage pushing through sharp pain.
- If Swapnil reports discomfort, suggest rest or a regression exercise.
- Always acknowledge emotions before giving instructions.

**CURRENT CONTEXT:**
Swapnil is recovering from ACL surgery. Be extremely patient, supportive, and motivating.
"""

# --- Global Mutable State (initialized at startup) ---
vector_store = None
groq_client = None
llm = None
