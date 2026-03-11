# 📖 Kineti-AI — Project Documentation

> **Version:** 1.0  
> **Last Updated:** February 17, 2026  
> **Author:** Kineti-AI Development Team  
> **Platform:** Android (Unity) + Python Backend  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Backend (Python / FastAPI)](#4-backend-python--fastapi)
   - 4.1 [main.py — Core Application Server](#41-mainpy--core-application-server)
   - 4.2 [rag_engine.py — Knowledge Base Builder](#42-rag_enginepy--knowledge-base-builder)
5. [Frontend (Unity / C#)](#5-frontend-unity--c)
   - 5.1 [HandsFreeVoice.cs — Main Client Controller](#51-handsfreevoicecs--main-client-controller)
   - 5.2 [KinetiClient.cs — Simple Query Client](#52-kineticlientcs--simple-query-client)
   - 5.3 [MediaPoseSolution.cs — Pose Data Manager](#53-mediaposesolutioncs--pose-data-manager)
   - 5.4 [MediaPoseVisualizer.cs — Skeleton Renderer](#54-mediaposevisualizercs--skeleton-renderer)
   - 5.5 [PoseDataSerializer.cs — Pose JSON Serializer](#55-posedataserializercs--pose-json-serializer)
   - 5.6 [ProMenuBuilder.cs — Main Menu UI](#56-promenubuildecs--main-menu-ui)
   - 5.7 [SessionSummary.cs — Session Results Screen](#57-sessionsummarycs--session-results-screen)
   - 5.8 [WavUtility.cs — WAV Audio Encoder](#58-wavutilitycs--wav-audio-encoder)
6. [API Reference](#6-api-reference)
7. [Data Flow & Communication Protocol](#7-data-flow--communication-protocol)
8. [State Machine — Conversation Manager](#8-state-machine--conversation-manager)
9. [Body Tracking Pipeline](#9-body-tracking-pipeline)
10. [RAG (Retrieval-Augmented Generation) Pipeline](#10-rag-retrieval-augmented-generation-pipeline)
11. [Setup & Installation](#11-setup--installation)
12. [Configuration](#12-configuration)
13. [Project Directory Structure](#13-project-directory-structure)
14. [Known Limitations & Future Work](#14-known-limitations--future-work)

---

## 1. Project Overview

**Kineti-AI** is an AI-powered virtual physical therapy assistant designed to guide patients through rehabilitation exercises — specifically **ACL (Anterior Cruciate Ligament) post-surgery recovery**. The system combines:

- **Real-time body tracking** via MediaPipe Pose Detection
- **Conversational AI** powered by Groq's LLM (LLaMA 3.1)
- **Medical knowledge retrieval** through RAG (Retrieval-Augmented Generation) using an ACL rehabilitation protocol PDF
- **Natural voice interaction** using Whisper for speech-to-text and Edge-TTS for text-to-speech
- **Exercise form analysis** with automated rep counting and real-time corrective feedback
- **Skeleton visualization** overlaid on the Unity AR camera feed

The application runs as a **client-server system**: an Android Unity app captures microphone audio and camera frames, sends them to a Python FastAPI backend server, which processes them and returns AI responses with audio and skeleton data.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ANDROID DEVICE                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Unity Application                     │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────┐ │  │
│  │  │ HandsFree   │  │ MediaPose    │  │ MediaPose│ │  │
│  │  │ Voice.cs    │──│ Solution.cs  │──│Visualizer│ │  │
│  │  │ (Main Loop) │  │ (Data Store) │  │ (GL Draw)│ │  │
│  │  └──────┬──────┘  └──────────────┘  └──────────┘ │  │
│  │         │                                          │  │
│  │    ┌────▼────┐   ┌───────────────┐                │  │
│  │    │ WavUtil │   │ PoseData      │                │  │
│  │    │ .cs     │   │ Serializer.cs │                │  │
│  │    └────┬────┘   └───────────────┘                │  │
│  │         │                                          │  │
│  │  ┌──────▼──────────────────┐                      │  │
│  │  │ HTTP POST /talk         │                      │  │
│  │  │ - Audio (WAV)           │                      │  │
│  │  │ - Camera Frame (Base64) │                      │  │
│  │  └──────┬──────────────────┘                      │  │
│  └─────────┼─────────────────────────────────────────┘  │
└────────────┼────────────────────────────────────────────┘
             │ WiFi / LAN
             ▼
┌─────────────────────────────────────────────────────────┐
│                   PYTHON BACKEND                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │              FastAPI Server (port 8000)             │  │
│  │                                                     │  │
│  │  ┌──────────────┐    ┌─────────────────────────┐  │  │
│  │  │  /speak      │    │  /talk                   │  │  │
│  │  │  (Greeting)  │    │  (Main Loop Endpoint)    │  │  │
│  │  └──────────────┘    └──────────┬──────────────┘  │  │
│  │                                  │                  │  │
│  │           ┌──────────────────────┼───────────┐     │  │
│  │           ▼                      ▼           ▼     │  │
│  │  ┌────────────────┐  ┌──────────────┐  ┌────────┐│  │
│  │  │ Whisper STT    │  │ MediaPipe    │  │ Groq   ││  │
│  │  │ (Groq API)     │  │ Pose Detect  │  │ LLM    ││  │
│  │  └────────────────┘  └──────────────┘  └────────┘│  │
│  │           │                   │            │      │  │
│  │           ▼                   ▼            ▼      │  │
│  │  ┌────────────────┐  ┌──────────────┐  ┌────────┐│  │
│  │  │ Conversation   │  │ Form         │  │ RAG    ││  │
│  │  │ State Machine  │  │ Analysis     │  │ Engine ││  │
│  │  └────────────────┘  └──────────────┘  └────────┘│  │
│  │           │                                │      │  │
│  │           ▼                                ▼      │  │
│  │  ┌──────────────────────────────────────────────┐│  │
│  │  │            Edge-TTS (Voice Synthesis)        ││  │
│  │  └──────────────────────────────────────────────┘│  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  ┌─────────────────┐  ┌───────────────────────────────┐ │
│  │ FAISS / ChromaDB│  │ ACL Protocol PDF (RAG Source) │ │
│  │ Vector Store    │  │ data/acl-protocol.pdf         │ │
│  └─────────────────┘  └───────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

### Backend
| Component | Technology | Purpose |
|---|---|---|
| Web Framework | **FastAPI** | Async HTTP server with multipart form support |
| LLM | **Groq API** (LLaMA 3.1 8B Instant) | Conversational AI responses |
| Speech-to-Text | **Whisper Large v3 Turbo** (via Groq) | Transcribe user voice commands |
| Text-to-Speech | **Edge-TTS** | Microsoft Azure neural voices (Andrew) |
| Pose Detection | **MediaPipe Pose Landmarker** (Tasks API) | Real-time body tracking from camera frames |
| RAG Embeddings | **HuggingFace** (all-MiniLM-L6-v2) | Sentence embeddings for document retrieval |
| Vector Store | **FAISS** / **ChromaDB** | Store and query document embeddings |
| Document Loader | **LangChain** (PyPDFLoader) | Parse ACL rehabilitation protocol PDF |
| Computer Vision | **OpenCV** (cv2) | Image decoding and color space conversion |
| Math/Arrays | **NumPy** | Numerical operations for pose processing |

### Frontend (Unity)
| Component | Technology | Purpose |
|---|---|---|
| Game Engine | **Unity 6** | Cross-platform AR application |
| Target Platform | **Android** | Mobile deployment with camera + mic |
| Networking | **UnityWebRequest** | HTTP communication with backend |
| UI Framework | **Unity UI** + **TextMesh Pro** | In-game HUD and text rendering |
| Audio | **Unity AudioSource** + **Microphone API** | Record and play audio |
| Camera | **WebCamTexture** | Capture camera frames for pose detection |
| Rendering | **GL.Lines** | Screen-space skeleton overlay rendering |

---

## 4. Backend (Python / FastAPI)

### 4.1 `main.py` — Core Application Server

**Location:** `Kineti-AI/backend/app/main.py`  
**Lines:** 696  
**Role:** The heart of the backend — handles all API endpoints, AI processing, pose detection, conversation management, and exercise analysis.

#### Key Components

##### Global State & Configuration
```python
# Landmark mapping for 12 tracked joints
LANDMARK_MAP = {
    11: "LeftShoulder",  12: "RightShoulder",
    13: "LeftElbow",     14: "RightElbow",
    15: "LeftWrist",     16: "RightWrist",
    23: "LeftHip",       24: "RightHip",
    25: "LeftKnee",      26: "RightKnee",
    27: "LeftAnkle",     28: "RightAnkle"
}
```

##### Session State Dictionary
The server maintains a mutable session dictionary that tracks the current therapy session:

| Key | Type | Description |
|---|---|---|
| `state` | `str` | Current conversation state: `check_in` → `propose` → `active_workout` |
| `pain_level` | `int` | User's reported pain (0-10) |
| `user_confirmed` | `bool` | Whether user explicitly confirmed starting an exercise |
| `paused_for_voice` | `bool` | Blocks vision processing while AI speaks |
| `active_exercise` | `str` | Current exercise name (e.g., "Squat") |
| `active_rules` | `dict` | Biomechanical rules from RAG extraction |
| `reps_count` | `int` | Number of completed repetitions |
| `target_reps` | `int` | Target rep count (default: 10) |
| `is_moving` | `bool` | Whether user is in the "down" phase of a rep |
| `last_response` | `str` | Last AI response text (for echo suppression) |
| `form_status` | `str` | Current form quality: `good` / `bad` / `neutral` |
| `current_angle` | `float` | Current knee angle in degrees |

##### Functions

| Function | Lines | Description |
|---|---|---|
| `init_mediapipe()` | 47–75 | Lazy initialization of MediaPipe Pose Landmarker using Tasks API with model file |
| `startup_event()` | 102–158 | FastAPI startup hook — initializes Groq client, LLM, persona, and RAG vector store |
| `detect_pose_from_frame(image_bytes)` | 164–235 | Decodes JPEG bytes, runs MediaPipe pose detection, returns backend + Unity coordinate JSON |
| `is_echo(user_text, last_text)` | 237–251 | Echo suppression using fuzzy matching (SequenceMatcher > 50% similarity) |
| `extract_exercise_rules(exercise_name)` | 253–280 | Queries RAG to extract biomechanical rules (joint, target angle, mode, cue) |
| `calculate_angle(p1, p2, p3)` | 282–293 | 3D angle calculation between three joint positions |
| `check_dynamic_rule(joints, rules)` | 295–313 | Validates if current skeleton meets exercise angle requirements |
| `analyze_squat_form(joints)` | 317–341 | Analyzes squat depth: good (<110°), bad (110-140°), neutral (>140°) |
| `process_vision_data(skeleton_json)` | 343–395 | Main vision processing with triple-check guards, rep counting, and form feedback |
| `manage_conversation(user_text)` | 397–540 | State machine handling all conversation flows across 3 states |

##### AI Persona
The backend uses a carefully crafted system prompt defining the AI as "KinetiAI" — a **warm, supportive physical therapist**:
- Celebrates small wins with encouragement
- Safety-first: never pushes through pain
- Concise responses during exercises (under 2 sentences)
- Never gives medical diagnoses
- Context: user is recovering from ACL surgery

---

### 4.2 `rag_engine.py` — Knowledge Base Builder

**Location:** `Kineti-AI/backend/app/rag_engine.py`  
**Lines:** 52  
**Role:** Standalone script to build the ChromaDB vector store from the ACL protocol PDF.

#### Functions

| Function | Description |
|---|---|
| `build_knowledge_base()` | Loads PDF → splits into 1000-char chunks (200 overlap) → embeds with MiniLM-L6-v2 → stores in ChromaDB |
| `get_retriever()` | Returns a retriever interface for querying the knowledge base (top-3 results) |

#### Usage
```bash
cd backend
python -m app.rag_engine  # Builds the knowledge base
```

> **Note:** The main server (`main.py`) uses FAISS at runtime for vector storage, while `rag_engine.py` uses ChromaDB as an alternative builder. Both use the same embedding model.

---

## 5. Frontend (Unity / C#)

### 5.1 `HandsFreeVoice.cs` — Main Client Controller

**Location:** `My project/Assets/Scripts/HandsFreeVoice.cs`  
**Lines:** 399  
**Role:** The primary Unity script orchestrating the entire client-side flow — microphone recording, camera capture, server communication, and audio playback.

#### Inspector Settings

| Field | Type | Default | Description |
|---|---|---|---|
| `serverUrl` | `string` | `http://192.168.1.9:8000` | Backend server IP address |
| `sensitivity` | `float` | `0.03` | Microphone voice activity detection threshold |
| `silenceTimeout` | `float` | `1.5s` | Seconds of silence before sending audio |
| `visionHeartbeatRate` | `float` | `2.0s` | Camera frame send interval during silent periods |

#### Core Loop (`VoiceLoop`)
```
┌────────────────────┐
│  Every Frame       │
│                    │
│  1. Check Mic Vol  │──── > sensitivity? ──→ Mark "hearing voice"
│                    │
│  2. Silence Check  │──── > 1.5s silence? ──→ Send Audio + Frame
│                    │
│  3. Vision Beat    │──── > 2.0s interval? ──→ Send Frame only
│                    │
└────────────────────┘
```

#### Communication Protocol
Sends HTTP POST to `/talk` with:
- `file`: WAV audio data (or tiny dummy heartbeat for vision-only)
- `skeleton_data`: JSON skeleton payload (fallback, `{}` when using camera)
- `camera_frame_b64`: Base64-encoded JPEG camera frame

Receives JSON response:
```json
{
  "status": "ok" | "silent",
  "feedback": "Great rep!",
  "rep_count": 5,
  "state": "active_workout",
  "landmarks": "{\"joints\": [...]}",
  "audio_b64": "<base64 MP3>",
  "form_status": "good" | "bad" | "neutral",
  "angle": 95.3
}
```

#### Key Methods

| Method | Description |
|---|---|
| `Start()` | Requests permissions, starts mic, camera, and sends greeting |
| `StartCamera()` | Initializes front-facing WebCamTexture at 640×480 @ 15fps |
| `CaptureFrameAsBase64()` | Captures camera frame as JPEG (quality 50) and encodes to Base64 |
| `SendGreeting()` | POST to `/speak` endpoint, plays greeting audio |
| `VoiceLoop()` | Continuous coroutine: detects speech, sends data, processes responses |
| `ProcessAndSend(sendAudio)` | Builds multipart form, sends to server, handles JSON response |
| `PlayAudioFromBytes(data)` | Saves MP3 to temp file, loads and plays via AudioSource |
| `GetAverageVolume()` | Calculates RMS volume from mic buffer for VAD |

---

### 5.2 `KinetiClient.cs` — Simple Query Client

**Location:** `My project/Assets/Scripts/KinetiClient.cs`  
**Lines:** 97  
**Role:** A simpler HTTP client for sending text-based queries to the backend. Used for testing/debug purposes (e.g., querying about Week 2 rehab goals).

---

### 5.3 `MediaPoseSolution.cs` — Pose Data Manager

**Location:** `My project/Assets/Scripts/MediaPoseSolution.cs`  
**Lines:** 136  
**Role:** Singleton that receives pose landmark data from the server and stores it in a `PoseLandmarkData` struct. Fires the `OnPoseUpdated` event for visualization.

#### Data Classes
```csharp
// 12 tracked joint positions
public class PoseLandmarkData {
    public Vector3 LeftShoulder, RightShoulder;
    public Vector3 LeftElbow, RightElbow;
    public Vector3 LeftWrist, RightWrist;
    public Vector3 LeftHip, RightHip;
    public Vector3 LeftKnee, RightKnee;
    public Vector3 LeftAnkle, RightAnkle;
    public float timestamp;
}

// JSON deserialization wrapper
public class JointDataWrapper { string name; float x, y, z; }
public class SkeletonPayloadWrapper { JointDataWrapper[] joints; }
```

#### Pose Timeout
If no pose data arrives for **3 seconds**, the pose is marked as lost.

---

### 5.4 `MediaPoseVisualizer.cs` — Skeleton Renderer

**Location:** `My project/Assets/Scripts/MediaPoseVisualizer.cs`  
**Lines:** 203  
**Role:** Renders the detected skeleton on screen using GL.Lines in orthographic projection. Supports smooth position interpolation and form-based color changes (green = good, red = bad).

#### Bone Connections
12 bone segments connecting:
```
Shoulders ←→ Shoulders
Shoulders → Elbows → Wrists
Shoulders → Hips
Hips ←→ Hips  
Hips → Knees → Ankles
```

#### Visual Features
- **Position smoothing** via `Vector3.Lerp` (speed: 8) to prevent jitter
- **Color interpolation** via `Color.Lerp` for smooth form feedback transitions
- **Joint dots** rendered as yellow quads
- **Multi-pass thick lines** (3 offset passes) for visibility
- **Auto-fade** after 3 seconds of no data

---

### 5.5 `PoseDataSerializer.cs` — Pose JSON Serializer

**Location:** `My project/Assets/Scripts/PoseDataSerializer.cs`  
**Lines:** 83  
**Role:** Converts stored `PoseLandmarkData` to JSON format matching the backend's expected schema. Serializes 8 key joints (hips, knees, ankles, shoulders) for exercise analysis.

---

### 5.6 `ProMenuBuilder.cs` — Main Menu UI

**Location:** `My project/Assets/Scripts/ProMenuBuilder.cs`  
**Lines:** 195  
**Role:** Programmatically generates the main menu UI at runtime with animations.

#### Features
- **Deep navy background** with medical teal accents
- **Animated title slide-down** with ease-out curve
- **"BEGIN SESSION" button** with elastic pop-in and continuous heartbeat pulse animation
- **Click flash effect** before scene transition
- Uses Unity 6's `LegacyRuntime.ttf` font

---

### 5.7 `SessionSummary.cs` — Session Results Screen

**Location:** `My project/Assets/Scripts/SessionSummary.cs`  
**Lines:** 83  
**Role:** Displays workout session results after completion. Shows total reps completed, feedback text, and a "FINISH" button that returns to the main menu.

---

### 5.8 `WavUtility.cs` — WAV Audio Encoder

**Location:** `My project/Assets/Scripts/WavUtility.cs`  
**Lines:** 88  
**Role:** Static utility that converts Unity's `AudioClip` to WAV byte array format with proper RIFF/WAVE headers. Used by `HandsFreeVoice.cs` to encode microphone recordings for the Whisper API.

#### WAV Format
- 16-bit PCM
- 16 kHz sample rate
- Mono channel
- Standard 44-byte WAV header

---

## 6. API Reference

### `POST /speak`
**Purpose:** Trigger the initial greeting  
**Request Body:** `{"text": "HELLO_INIT"}`  
**Response:** MP3 audio file (greeting message)  
**Side Effect:** Resets session state to `check_in`

### `POST /talk`
**Purpose:** Main communication loop  
**Request:** Multipart form data:

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | WAV audio file (can be tiny dummy for vision-only) |
| `skeleton_data` | String | No | JSON skeleton data (fallback) |
| `camera_frame_b64` | String | No | Base64-encoded JPEG camera frame |

**Response:** JSON

| Field | Type | Description |
|---|---|---|
| `status` | string | `"ok"` (has response) or `"silent"` (no response needed) |
| `feedback` | string | AI feedback text |
| `rep_count` | int | Current rep count |
| `state` | string | Current session state |
| `landmarks` | string | JSON skeleton for Unity visualization |
| `audio_b64` | string | Base64-encoded MP3 audio response |
| `form_status` | string | `"good"`, `"bad"`, or `"neutral"` |
| `angle` | float | Current knee angle in degrees |

---

## 7. Data Flow & Communication Protocol

```
Unity App                              Python Backend
─────────                              ─────────────
1. Start App
   ├─ Init Mic (16kHz)
   ├─ Init Camera (640×480, 15fps)
   └─ POST /speak ──────────────────→  Generate greeting TTS
                                        ← MP3 audio ──────────
2. Main Loop (every frame)
   ├─ Detect voice activity
   │   └─ If speaking → buffer audio
   │   └─ If 1.5s silence after speech:
   │       ├─ Capture camera frame → JPEG → Base64
   │       ├─ Encode mic buffer → WAV
   │       └─ POST /talk ────────────────→  Step 0: MediaPipe Pose Detection
   │                                         Step 1: Whisper Transcription
   │                                         Step 2: Conversation Manager
   │                                         Step 3: Vision Processing
   │                                         Step 4: Edge-TTS Synthesis
   │                                        ← JSON response ──────────
   │       ├─ Update skeleton visualization
   │       ├─ Update UI (reps, feedback)
   │       └─ Play response audio
   │
   └─ If >2.5s since last speech AND >2s since last vision:
       ├─ Capture camera frame → Base64
       ├─ Send dummy WAV (44 bytes)
       └─ POST /talk (vision heartbeat) ──→  Pose detection + vision only
                                            ← JSON response ──────────
```

---

## 8. State Machine — Conversation Manager

The backend implements a 3-state conversation flow:

```
                              ┌──────────────────┐
              ┌──────────────→│   CHECK_IN       │◄──── Pain / Stop
              │               │  "How do you     │      during workout
              │               │   feel today?"   │
              │               └────────┬─────────┘
              │                        │
              │               Assess pain level
              │               (NLP: good/medium/pain)
              │                        │
              │                        ▼
              │               ┌──────────────────┐
              │               │   PROPOSE        │
  Workout     │               │  "Let's do       │
  Complete    │               │   Squats. Ready?" │
  / Pain      │               └────────┬─────────┘
              │                        │
              │               Explicit "yes"/"ready"
              │               (Gatekeeper check)
              │                        │
              │                        ▼
              │               ┌──────────────────┐
              └───────────────│  ACTIVE_WORKOUT  │
                              │  - Rep counting  │
                              │  - Form feedback │
                              │  - Voice priority│
                              └──────────────────┘
```

### State 1: `check_in`
- Assesses user pain level using NLP
- Handles negation: "no pain" → positive, "pain" → negative
- Sets pain_level which determines exercise difficulty

### State 2: `propose`
- Suggests exercise based on pain level (Squats for low pain, Heel Slides for high)
- **Critical gatekeeper**: Only explicit "yes"/"ready"/"start" advances to workout
- Unclear input gets: "I need a clear 'yes' before we start"

### State 3: `active_workout`
- Voice commands take priority over vision processing
- Safety: any pain keyword immediately stops workout
- Stop commands end session and show rep summary

---

## 9. Body Tracking Pipeline

```
Camera Frame (640×480 JPEG, Quality 50)
         │
         ▼
   Base64 Encode (Unity)
         │
         ▼
   HTTP POST /talk (camera_frame_b64 field)
         │
         ▼
   Base64 Decode (Python)
         │
         ▼
   numpy.frombuffer → cv2.imdecode (JPEG → BGR)
         │
         ▼
   cv2.cvtColor (BGR → RGB)
         │
         ▼
   MediaPipe PoseLandmarker.detect()
   ├─ Model: pose_landmarker.task (30.6 MB)
   ├─ Mode: IMAGE (single frame)
   ├─ Min detection confidence: 0.3
   └─ Min tracking confidence: 0.3
         │
         ▼
   Extract 12 landmarks (from 33 total)
         │
    ┌────┴────┐
    ▼         ▼
 Backend    Unity
 Coords     Coords
 (raw)      (Y flipped: 1.0 - y)
    │         │
    ▼         ▼
 Squat      JSON → Unity
 Analysis   MediaPoseSolution
    │         │
    ▼         ▼
 Rep Count  MediaPoseVisualizer
 + Form     (GL.Lines skeleton)
 Feedback
```

### Tracked Joints (12 of 33 MediaPipe landmarks)

| Index | Joint Name | Body Part |
|---|---|---|
| 11 | LeftShoulder | Upper body |
| 12 | RightShoulder | Upper body |
| 13 | LeftElbow | Arms |
| 14 | RightElbow | Arms |
| 15 | LeftWrist | Arms |
| 16 | RightWrist | Arms |
| 23 | LeftHip | Lower body |
| 24 | RightHip | Lower body |
| 25 | LeftKnee | Legs |
| 26 | RightKnee | Legs |
| 27 | LeftAnkle | Legs |
| 28 | RightAnkle | Legs |

### Squat Form Analysis

| Knee Angle | Form Status | Feedback |
|---|---|---|
| < 110° | ✅ Good | "Great depth!" |
| 110°–140° | ❌ Bad | "Go deeper! Bend your knees more." |
| > 140° | ⚪ Neutral | Standing position (no feedback) |
| > 155° (after down) | Rep Counted | "Rep N. Keep going!" |

### Rep Counting Algorithm
1. **Down phase:** Knee angle drops below 110° → `is_moving = True`
2. **Up phase:** Knee angle rises above 155° after being down → Rep counted, `is_moving = False`
3. **Milestone feedback:** Rep 5 → "Halfway there!", Rep 10 → "Set complete!"

---

## 10. RAG (Retrieval-Augmented Generation) Pipeline

```
ACL Protocol PDF (data/acl-protocol.pdf)
         │
         ▼
   PyPDF Loader (LangChain)
         │
         ▼
   Recursive Text Splitter
   (chunk_size=1000, overlap=100-200)
         │
         ▼
   HuggingFace Embeddings
   (sentence-transformers/all-MiniLM-L6-v2)
         │
         ▼
   FAISS Vector Store (runtime)
   ChromaDB (builder script, chroma_db/)
         │
         ▼
   Retriever (top-k=2-3)
         │
         ▼
   Context injection into LLM prompt
   + Exercise rule extraction
```

### Exercise Rule Extraction
When a workout starts, the system queries RAG for biomechanical rules:

```json
{
  "joint": "RightKnee",
  "target": 90,
  "mode": "min",
  "cue": "Keep your back straight"
}
```

- `joint`: Which joint to measure
- `target`: Target angle in degrees
- `mode`: `"min"` (angle should go below target) or `"max"` (above)
- `cue`: Verbal coaching cue for the user

---

## 11. Setup & Installation

### Prerequisites
- **Python 3.10+**
- **Unity 6** (with Android Build Support)
- **Android device** with camera + microphone
- **Groq API Key** (free at groq.com)

### Backend Setup
```bash
# 1. Navigate to backend
cd "FInal Year Project/Kineti-AI/backend"

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
# Create .env file with:
# GROQ_API_KEY=your_groq_api_key_here

# 5. Build knowledge base (one-time)
python -m app.rag_engine

# 6. Download MediaPipe model (if not present)
# Place pose_landmarker.task in backend/ directory
# Download from: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

# 7. Start server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Unity Setup
1. Open `My project` in Unity 6
2. Set build target to **Android** (File → Build Settings)
3. In `HandsFreeVoice.cs` Inspector:
   - Set `serverUrl` to your PC's local IP (e.g., `http://192.168.1.9:8000`)
4. Ensure scenes are in Build Settings:
   - `MainMenu` (Scene 0)
   - `SampleScene` (Scene 1)
5. Build and deploy to Android device
6. Ensure phone and PC are on the **same WiFi network**

---

## 12. Configuration

### Environment Variables (`.env`)
| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | API key for Groq (LLM + Whisper) |

### Backend Constants (in `main.py`)
| Constant | Value | Description |
|---|---|---|
| `PDF_FILE_PATH` | `"data/acl-protocol.pdf"` | Path to the rehabilitation protocol PDF |
| `VOICE_NAME` | `"en-US-AndrewMultilingualNeural"` | Edge-TTS voice (natural human-like) |
| Min detection confidence | `0.3` | MediaPipe pose detection threshold |
| Min tracking confidence | `0.3` | MediaPipe pose tracking threshold |
| Echo suppression threshold | `0.5` | SequenceMatcher similarity ratio for echo detection |

### Unity Inspector Settings
| Parameter | Default | Tuning |
|---|---|---|
| `sensitivity` | `0.03` | Lower = more sensitive mic, higher = less noise |
| `silenceTimeout` | `1.5s` | Time after last sound before sending |
| `visionHeartbeatRate` | `2.0s` | How often to send camera-only frames |
| Camera resolution | `640×480` | Balance quality vs bandwidth |
| JPEG quality | `50` | Balance quality vs upload size |
| Skeleton lerp speed | `8` | Higher = faster tracking, more jitter |

---

## 13. Project Directory Structure

```
FInal Year Project/
├── .env                          # Environment variables (GROQ_API_KEY)
│
├── Kineti-AI/                    # Backend repository
│   └── backend/
│       ├── app/
│       │   ├── main.py           # Core FastAPI server (696 lines)
│       │   └── rag_engine.py     # ChromaDB knowledge base builder (52 lines)
│       ├── data/
│       │   └── acl-protocol.pdf  # ACL rehabilitation protocol (RAG source)
│       ├── chroma_db/            # ChromaDB vector store (generated)
│       ├── pose_landmarker.task  # MediaPipe model file (30.6 MB)
│       ├── requirements.txt     # Python dependencies
│       ├── greeting.mp3          # Cached greeting audio
│       ├── response.mp3          # Latest response audio
│       ├── user_data.json        # User session data
│       └── venv/                 # Python virtual environment
│
├── My project/                   # Unity project
│   ├── Assets/
│   │   ├── Scripts/
│   │   │   ├── HandsFreeVoice.cs       # Main client controller (399 lines)
│   │   │   ├── KinetiClient.cs         # Simple query client (97 lines)
│   │   │   ├── MediaPoseSolution.cs    # Pose data singleton (136 lines)
│   │   │   ├── MediaPoseVisualizer.cs  # GL skeleton renderer (203 lines)
│   │   │   ├── PoseDataSerializer.cs   # Pose JSON serializer (83 lines)
│   │   │   ├── ProMenuBuilder.cs       # Main menu UI builder (195 lines)
│   │   │   ├── SessionSummary.cs       # Session results screen (83 lines)
│   │   │   └── WavUtility.cs           # WAV audio encoder (88 lines)
│   │   ├── Scenes/               # Unity scenes (MainMenu, SampleScene)
│   │   ├── Prefabs/              # BodyTracker, Line, Red prefabs
│   │   ├── Materials/            # SkeletonGlow, SkeletonMaterial, Red
│   │   └── XR/                   # XR/AR Foundation settings
│   ├── Packages/                 # Unity package manifest
│   └── ProjectSettings/         # Unity project configuration
│
└── builds/                       # Android APK builds
```

---

## 14. Known Limitations & Future Work

### Current Limitations
1. **Single-user session:** The backend uses a global session dictionary — no multi-user support
2. **Same-network requirement:** Unity client must be on the same WiFi as the backend server
3. **Limited exercise library:** Currently only Squats and Heel Slides are fully implemented
4. **No persistent data:** Session data resets on server restart
5. **Echo suppression trade-off:** 50% similarity threshold may occasionally suppress valid user speech that sounds similar to recent AI output
6. **Camera frame bottleneck:** Base64 encoding of camera frames adds latency

### Future Work
- **Multi-exercise support** with dynamic rule extraction from RAG
- **Session persistence** using a database for long-term progress tracking
- **WebSocket communication** for lower-latency real-time updates
- **On-device pose detection** using MediaPipe for Unity to reduce server dependency
- **Multi-user support** with authentication and individual session management
- **Extended pain assessment** with formal outcome measures (VAS, KOOS scores)
- **Exercise progression algorithms** that adapt difficulty over time
- **Data export** for healthcare provider review

---

> *This documentation was generated for the Kineti-AI Final Year Project.*
