# 📋 Kineti-AI — Project Review

> **Review Date:** February 17, 2026  
> **Project:** Kineti-AI — AI-Powered Virtual Physical Therapy Assistant  
> **Platform:** Android (Unity 6) + Python (FastAPI) Backend  
> **Domain:** Healthcare / Rehabilitation Technology  

---

## Overall Assessment

| Criteria | Rating | Grade |
|---|:---:|:---:|
| **Innovation & Concept** | ★★★★★ | A |
| **System Architecture** | ★★★★☆ | B+ |
| **Code Quality** | ★★★☆☆ | B |
| **Feature Completeness** | ★★★★☆ | B+ |
| **User Experience Design** | ★★★★☆ | A- |
| **Safety & Ethics** | ★★★★★ | A |
| **Technical Integration** | ★★★★★ | A |
| **Scalability** | ★★☆☆☆ | C |
| **Documentation** | ★★★★★ | A |
| **Overall** | **★★★★☆** | **B+** |

---

## 1. Innovation & Concept — ★★★★★

### Strengths
- **Highly relevant problem space:** ACL rehabilitation is a common post-surgical need with high dropout rates; an AI assistant addresses real clinical gaps.
- **Multi-modal AI integration:** Combining voice + vision + medical knowledge retrieval in one coherent system is technically ambitious and well-executed.
- **Hands-free design philosophy:** The entire interaction model avoids touch input, which is critical for a rehabilitation context where users are exercising.
- **Safety-first persona design:** The AI persona is thoughtfully crafted with medical ethics guardrails — it never pushes through pain, never diagnoses, and always prioritizes patient safety.

### Impact Potential
This project demonstrates a viable prototype for AI-assisted telerehabilitation. The combination of real-time body tracking, conversational coaching, and evidence-based exercise protocols positions it as a meaningful contribution to healthcare accessibility.

---

## 2. System Architecture — ★★★★☆

### Strengths
- **Clean client-server separation:** Unity handles I/O (camera, mic, display), while Python handles all AI/ML processing — a sensible division of labor.
- **Stateful conversation management:** The 3-state machine (`check_in` → `propose` → `active_workout`) is well-designed with proper guards against premature state transitions.
- **Multi-pipeline processing:** Audio transcription, pose detection, and LLM response generation are logically separated within the `/talk` endpoint.
- **Echo suppression system:** Using fuzzy string matching (SequenceMatcher) to prevent the AI from responding to its own speech output is a practical solution.

### Areas for Improvement
- **Global session state:** Using a module-level dictionary (`current_session`) for session state is a single-point-of-failure and prevents multi-user support. Recommendation: Use a session manager with client IDs.
- **Synchronous pose detection:** MediaPipe runs synchronously on the server, which could block the event loop. Consider offloading to a thread pool or using async execution.
- **No WebSocket support:** The HTTP polling model (heartbeat every 2 seconds) introduces unnecessary latency. WebSockets would provide true real-time communication.
- **Tight coupling in `/talk` endpoint:** The `/talk` function (134 lines) handles audio transcription, pose detection, conversation management, form analysis, AND TTS synthesis. This could be refactored into smaller middleware functions.

---

## 3. Code Quality — ★★★☆☆

### Strengths
- **Descriptive comments:** Functions have clear docstrings explaining purpose and return values.
- **Section organization:** Code is organized with numbered section headers (`--- 1. INITIALIZATION ---`, `--- 2. LOGIC HELPERS ---`, etc.).
- **Emoji logging:** The use of emoji prefixes in log messages (`✅`, `⚠️`, `🗣️`, `🦴`) makes debugging visually intuitive.
- **Defensive coding:** Multiple null checks, try/except blocks, and graceful fallback behavior.

### Areas for Improvement

#### Backend (`main.py`)
- **File is too large (696 lines):** `main.py` contains everything — models, configuration, business logic, endpoints, and utility functions. This should be split into modules:
  ```
  app/
  ├── main.py          # App setup + endpoints only
  ├── models.py        # Pydantic models
  ├── session.py       # Session state management
  ├── conversation.py  # State machine logic
  ├── pose_engine.py   # MediaPipe detection + analysis
  ├── voice_engine.py  # TTS + STT handling
  └── config.py        # Configuration constants
  ```
- **Magic numbers:** Squat thresholds (110°, 140°, 155°) and echo suppression (0.5 ratio) should be named constants.
- **Error handling:** Some `except` blocks use bare `except:` without specifying exception types (lines 292, 307), which swallows all errors silently.
- **No type hints:** Function parameters and return types lack type annotations, reducing IDE support and readability.
- **Hardcoded voice name:** The TTS voice (`en-US-AndrewMultilingualNeural`) is hardcoded in a constant but should be configurable via environment variables.

#### Frontend (Unity C#)
- **Coroutine-heavy architecture:** The Unity scripts heavily rely on coroutines for async operations, which is acceptable but makes error handling and flow control harder to trace.
- **String-based JSON parsing:** Using `JsonUtility.FromJson<T>()` with string checks (`response.landmarks.Length > 10`) is fragile. Consider using a more robust JSON library.
- **Hardcoded IP address:** `serverUrl` defaults to `192.168.1.9:8000` — should be configurable or use a discovery mechanism.

---

## 4. Feature Completeness — ★★★★☆

### Implemented Features ✅
| Feature | Status | Quality |
|---|:---:|---|
| Voice-based conversational AI | ✅ | Excellent — natural persona, context-aware |
| Speech-to-text (Whisper) | ✅ | High quality via Groq API |
| Text-to-speech (Edge-TTS) | ✅ | Natural-sounding (Andrew voice) |
| Real-time body tracking | ✅ | MediaPipe Tasks API with 12 landmarks |
| Skeleton visualization | ✅ | GL-rendered with color feedback |
| Squat form analysis | ✅ | Angle-based with good/bad/neutral |
| Automatic rep counting | ✅ | Down/up phase detection |
| Pain assessment | ✅ | NLP with negation handling |
| Medical knowledge RAG | ✅ | ACL protocol PDF querying |
| Exercise rule extraction | ✅ | LLM + RAG auto-extraction |
| Echo suppression | ✅ | Fuzzy matching prevents loops |
| Main menu with animations | ✅ | Professional-looking animated UI |
| Session summary screen | ✅ | Shows reps + feedback |
| Camera preview option | ✅ | Optional RawImage display |

### Missing / Incomplete Features ❌
| Feature | Status | Impact |
|---|:---:|---|
| Multiple exercise types | ⚠️ Partial | Only Squats fully implemented, Heel Slides mentioned but not tracked |
| Progress history | ❌ | No persistent data across sessions |
| User authentication | ❌ | No user accounts or profiles |
| Bilateral tracking | ⚠️ Partial | Both knees averaged but no individual feedback |
| Network error recovery | ⚠️ Basic | Limited retry logic, no offline mode |
| Settings UI | ❌ | Server URL and sensitivity only configurable in Inspector |
| Exercise demonstration | ❌ | No visual guide showing correct form |

---

## 5. User Experience Design — ★★★★☆

### Strengths
- **Hands-free operation:** Users never need to touch the screen during exercises — voice + vision handles everything.
- **Natural conversation flow:** The AI doesn't feel like a chatbot; it follows a logical therapy session structure (check-in → propose → workout).
- **Real-time visual feedback:** Skeleton color changes from green to red provide immediate, intuitive form feedback without reading text.
- **Progressive encouragement:** Milestone messages at rep 5 ("Halfway there!") and rep 10 ("Set complete!") maintain motivation.
- **Safety gatekeeper:** The system explicitly requires user confirmation before starting exercises, preventing accidental workout starts.
- **Polished menu screen:** The animated main menu with heartbeat button creates a premium feel.

### Areas for Improvement
- **No onboarding tutorial:** First-time users have no guidance on positioning, camera distance, etc.
- **Feedback delay:** The 2-second heartbeat rate means visual feedback can lag by up to 2 seconds.
- **Audio gap during processing:** There's a noticeable pause between user speech and AI response while audio processes.
- **No progress visualization:** Users can't see their historical performance or improvement over time.

---

## 6. Safety & Ethics — ★★★★★

### Exemplary Practices
1. **Pain-responsive behavior:** Any mention of pain during exercise immediately stops the workout and switches to assessment mode.
2. **No medical diagnosis:** The persona explicitly forbids statements like "You have a tear" — it only provides exercise guidance.
3. **No "push through" mentality:** The AI persona is deliberately designed to never encourage exercising through sharp pain.
4. **Emotional acknowledgment:** The system acknowledges user emotions before giving instructions, following therapeutic communication best practices.
5. **Explicit consent model:** Workout never starts without clear user confirmation (the "Gatekeeper" pattern).
6. **Evidence-based protocols:** Exercise rules are extracted from an actual ACL rehabilitation protocol document, not arbitrary values.

### Recommendations
- Add a medical disclaimer on app startup
- Implement emergency contact information access
- Consider adding a "report to healthcare provider" feature for sessions with high pain levels

---

## 7. Technical Integration — ★★★★★

### AI/ML Component Integration
The project demonstrates exceptional ability to integrate multiple AI/ML services into a cohesive product:

| Component | Integration Quality | Notes |
|---|---|---|
| MediaPipe → Backend | Excellent | Clean JPEG → numpy → MediaPipe pipeline |
| Whisper → Conversation | Excellent | File-size gating prevents unnecessary transcription costs |
| RAG → Exercise Rules | Good | Dynamic extraction with JSON schema enforcement |
| LLM → Response Gen | Excellent | Context-aware with persona injection |
| Edge-TTS → Client | Excellent | Base64-encoded MP3 in JSON response is elegant |
| Unity ↔ Backend | Good | Multipart form for mixed data types works well |

### Coordinate System Handling
The Y-axis flip for Unity coordinates (`1.0 - y`) in the backend shows proper understanding of the MediaPipe-to-Unity coordinate mapping.

---

## 8. Scalability — ★★☆☆☆

### Critical Concerns
1. **Single-session global state:** `current_session` is a module-level dictionary — two users would overwrite each other's state.
2. **File-based audio:** TTS saves to `response.mp3` / `greeting.mp3` in the working directory — concurrent requests would overwrite each other.
3. **No rate limiting:** The `/talk` endpoint is unprotected against rapid-fire requests.
4. **Synchronous model loading:** MediaPipe and FAISS load on startup, blocking the server.
5. **No database:** Session data, user data, and progress are all in-memory or flat files.

### Recommendations for Production
- Use Redis or a database for session management with unique client IDs
- Use temporary file paths with unique identifiers for audio files
- Add connection pooling and request queuing
- Implement user authentication with JWT tokens
- Deploy behind a reverse proxy (nginx) with rate limiting
- Use Docker for consistent deployment

---

## 9. Security Assessment

### Current Risks
| Risk | Severity | Description |
|---|:---:|---|
| No HTTPS | High | Audio and medical data transmitted in plaintext |
| No authentication | High | Anyone on the network can access the API |
| No input validation | Medium | `camera_frame_b64` and `skeleton_data` not size-limited |
| File system access | Medium | `cv2.imwrite` for debug frames could be exploited |
| API key in .env | Low | GROQ_API_KEY stored in plaintext (standard practice but note it) |

### Recommendations
- Add TLS/HTTPS support (use nginx as reverse proxy)
- Implement API key or token authentication
- Add request size limits (max 5MB per frame)
- Remove debug frame saving in production
- Use environment-specific configuration (dev vs prod)

---

## 10. Testing & Quality Assurance

### Current State
- **No automated tests** found in the codebase
- **No CI/CD pipeline** configured
- **Manual testing** appears to be the primary validation method
- **Debug logging** is comprehensive (emoji-prefixed logs)

### Recommendations
1. **Unit tests** for:
   - `calculate_angle()` — verify angle calculation accuracy
   - `is_echo()` — test echo suppression edge cases
   - `manage_conversation()` — test all state transitions
   - `analyze_squat_form()` — test threshold boundaries
2. **Integration tests** for `/talk` endpoint with mock audio + frames
3. **Performance benchmarks** for MediaPipe detection latency
4. **Unity Play Mode tests** for coroutine flows

---

## 11. Performance Analysis

### Identified Bottlenecks
| Operation | Estimated Time | Priority |
|---|---|---|
| Base64 encode/decode (camera frame) | ~10-30ms | Medium |
| MediaPipe pose detection | ~50-150ms | High |
| Whisper transcription (Groq API) | ~300-800ms | High |
| LLM response generation (Groq API) | ~200-600ms | Medium |
| Edge-TTS synthesis | ~100-500ms | Medium |
| Network round-trip (WiFi) | ~20-100ms | Low |
| **Total (with voice)** | **~680-2,180ms** | - |
| **Total (vision only)** | **~80-280ms** | - |

### Optimization Suggestions
- Stream TTS audio while generating (avoid waiting for full file)
- Use WebSockets to eliminate HTTP overhead
- Batch pose detection + form analysis in parallel with transcription
- Cache LLM responses for repeated scenarios (e.g., rep count messages)
- Reduce camera resolution to 320×240 for faster processing
- Consider on-device pose detection (MediaPipe for Unity SDK)

---

## 12. Comparative Analysis

### vs. Commercial Solutions

| Feature | Kineti-AI | SWORD Health | Kaia Health |
|---|:---:|:---:|:---:|
| AI Conversation | ✅ Voice | ❌ Text only | ❌ Pre-recorded |
| Body Tracking | ✅ MediaPipe | ✅ Proprietary | ✅ Proprietary |
| Rep Counting | ✅ | ✅ | ✅ |
| Form Feedback | ✅ Real-time | ✅ Post-exercise | ✅ Real-time |
| Medical Knowledge | ✅ RAG | ✅ Clinical team | ✅ Guidelines |
| Hands-Free | ✅ Voice | ❌ | ❌ |
| Cost | Free/Open | $$$$ | $$$ |

### Unique Differentiators
1. **Fully voice-controlled** — no commercial solution offers conversational AI coaching during exercises
2. **RAG-powered medical knowledge** — dynamically queries rehabilitation protocols rather than hard-coded advice
3. **Open-source potential** — can be adapted for any rehabilitation protocol by swapping the PDF

---

## 13. Final Verdict

### Summary
Kineti-AI is a **technically impressive final year project** that successfully integrates five distinct AI/ML technologies (pose detection, speech recognition, language model, text-to-speech, RAG) into a cohesive healthcare application. The safety-first design philosophy and hands-free interaction model demonstrate sophisticated understanding of the clinical use case.

### Key Achievements
✅ Real-time body tracking with form analysis and rep counting  
✅ Natural conversational AI with stateful therapy session management  
✅ Evidence-based exercise guidance via RAG integration  
✅ Polished user experience with animated UI and audio feedback  
✅ Comprehensive safety guardrails (pain detection, consent gating)

### Critical Recommendations (Priority Order)
1. **Refactor `main.py` into modules** — Single-file backend is the most urgent code quality issue
2. **Add session management** — Replace global dict with proper session handling for multi-user support
3. **Add automated tests** — Critical for healthcare software reliability
4. **Implement HTTPS** — Medical data must be transmitted securely
5. **Expand exercise library** — Current squat-only tracking limits clinical utility

### Final Grade: **B+ (Excellent for Final Year Project)**

The project demonstrates strong technical skills across AI/ML, mobile development, and system design. While it requires production hardening (security, scalability, testing), the core concept, architecture, and user experience are well above average for an academic project. The healthcare domain understanding and safety-conscious design are particularly commendable.

---

> *Review generated on February 17, 2026*
