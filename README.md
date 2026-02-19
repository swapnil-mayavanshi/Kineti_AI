# KinetiAI 🦾
**Real-time AI Physical Therapy Assistant for ACL Rehabilitation**

KinetiAI is an interactive, voice-driven, and vision-enabled AI assistant designed to help patients safely complete physical therapy exercises. By combining real-time skeleton tracking with a conversational AI "Physical Therapist" persona, KinetiAI guides users through their recovery protocols safely and effectively.

## ✨ Key Features
* **🗣️ Hands-Free Voice Interaction:** Completely voice-driven interface using Groq's Whisper for speech-to-text and Edge-TTS for low-latency, natural voice generation.
* **👁️ Real-Time Vision Tracking:** Analyzes user biomechanics using AR skeleton data. It calculates joint angles (e.g., knee flexion) dynamically and tracks repetitions only when perfect form is achieved.
* **🧠 Clinical RAG Brain:** Integrates a Retrieval-Augmented Generation (RAG) system loaded with ACL rehabilitation protocols to ensure the AI's exercise recommendations and target angles are medically safe.
* **🔄 Smart Session Management:** Uses a rigid state machine (`Check-in` ➡️ `Propose Exercise` ➡️ `Active Workout`) to prevent premature workouts and ensure the user is physically ready.
* **🔇 Smart Echo Cancellation:** Custom `difflib`-based audio filtering prevents the AI from hearing its own voice through the user's microphone.

## 🛠️ Tech Stack

**Backend (AI & Logic)**
* **Python / FastAPI:** High-performance async server routing vision and audio data.
* **Groq API:** Powers the core intelligence with `llama-3.1-8b-instant` and lightning-fast transcription with `whisper-large-v3-turbo`.
* **LangChain & FAISS:** Handles document chunking and vector search for the ACL PDF protocol.
* **Edge-TTS:** Generates the Physical Therapist's voice.

**Frontend (Client)**
* **Unity 3D / C#:** Handles the user interface, microphone capture, and audio playback.
* **AR Foundation / Body Tracking:** Captures real-time joint coordinates (x, y, z) and transmits them to the backend via REST API.
* **Android target:** Optimized for mobile deployment with necessary Camera and Mic permissions.

## 🚀 Getting Started

### Prerequisites
1. Python 3.10+
2. Unity 2022+ (with Android Build Support)
3. A Groq API Key

### 1. Backend Setup
Clone the repository and install the Python dependencies:
```bash
git clone [https://github.com/yourusername/KinetiAI.git](https://github.com/yourusername/KinetiAI.git)
cd KinetiAI/backend
pip install fastapi uvicorn pydantic python-dotenv groq langchain-groq langchain-huggingface langchain-community faiss-cpu pypdf edge-tts
