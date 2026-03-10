"""
Kineti-AI — Main Application
FastAPI app setup and endpoint definitions.
All business logic is in separate modules.
"""

import os
import shutil
import base64

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse

from app.config import GROQ_API_KEY, PDF_FILE_PATH, SYSTEM_INSTRUCTION, VOICE_NAME, MIN_AUDIO_SIZE_BYTES
from app import config
from app.models import TextRequest
from app import session
from app.pose_engine import detect_pose_from_frame, process_vision_data
from app.voice_engine import is_echo, synthesize_speech
from app.conversation import manage_conversation

# --- App ---
app = FastAPI()
_vision_count = 0
_debug_poses = []  # Store last 5 poses for debugging


# --- DEBUG ENDPOINT ---
@app.get("/debug")
def debug():
    """Returns last 5 detected pose snapshots so you can verify coordinates change."""
    cs = session.current_session
    return {
        "session_state": cs["state"],
        "reps_count": cs["reps_count"],
        "is_moving": cs["is_moving"],
        "form_status": cs.get("form_status", "neutral"),
        "current_angle": round(cs.get("current_angle", 180), 1),
        "last_poses_count": len(_debug_poses),
        "last_poses": _debug_poses[-5:]  # Last 5 snapshots
    }


# --- CONNECTIVITY TEST ---
@app.get("/ping")
def ping():
    """Simple connectivity test — visit http://YOUR_IP:8000/ping from phone browser."""
    return {"status": "ok", "message": "Kineti-AI server is reachable!"}


@app.get("/status")
def status():
    """Server status check."""
    return {
        "server": "running",
        "groq": config.groq_client is not None,
        "rag": config.vector_store is not None,
        "session_state": session.current_session["state"]
    }


# --- 1. INITIALIZATION ---
@app.on_event("startup")
def startup_event():
    print("⚡ Initializing Kineti-AI (Optimized Backend)...")

    # 1. Setup Groq
    if GROQ_API_KEY:
        from groq import Groq
        config.groq_client = Groq(api_key=GROQ_API_KEY)

        # Import ChatGroq — resolve Pydantic v2 forward-ref issues
        # by importing all required types before instantiation
        from langchain_core.caches import BaseCache  # noqa: F401
        from langchain_core.callbacks import Callbacks  # noqa: F401
        from langchain_groq import ChatGroq
        ChatGroq.model_rebuild()
        config.llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0.1)

    # 2. Setup RAG
    if os.path.exists(PDF_FILE_PATH):
        print(f"📚 Loading PDF Brain: {PDF_FILE_PATH}...")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            loader = PyPDFLoader(PDF_FILE_PATH)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)
            config.vector_store = FAISS.from_documents(splits, embeddings)
            print("✅ Brain Ready.")
        except Exception as e:
            print(f"⚠️ RAG Load Failed: {e}")
    else:
        print("⚠️ Warning: PDF not found. RAG disabled.")


# --- 2. ENDPOINTS ---

@app.get("/start_workout")
def start_workout():
    """Quick-start workout mode — enables rep counting and form analysis immediately."""
    cs = session.current_session
    cs["state"] = "active_workout"
    cs["reps_count"] = 0
    cs["is_moving"] = False
    cs["form_status"] = "neutral"
    cs["current_angle"] = 180
    print("🏋️ Workout quick-started via /start_workout")
    return {"status": "workout_started", "message": "Rep counting and form analysis are now active!"}


@app.post("/vision")
async def vision(
    camera_frame_b64: str = Form(""),
    rotation: str = Form("0"),
    mirrored: str = Form("0")
):
    """
    Lightweight vision-only endpoint — no audio processing.
    Receives camera frame + rotation info, returns landmarks + pose status.
    """
    global _vision_count
    _vision_count += 1
    cs = session.current_session
    response_data = {
        "landmarks": "{}",
        "pose_detected": False,
        "form_status": cs.get("form_status", "neutral"),
        "angle": round(cs.get("current_angle", 180), 1),
        "rep_count": cs["reps_count"],
        "feedback": ""
    }

    if camera_frame_b64 and len(camera_frame_b64) > 100:
        try:
            frame_bytes = base64.b64decode(camera_frame_b64)
            rot_angle = int(rotation)
            is_mirrored = (mirrored == "1")
            result = detect_pose_from_frame(frame_bytes, rot_angle, is_mirrored)
            if isinstance(result, tuple):
                skeleton_data, unity_landmarks = result
                if unity_landmarks != "{}":
                    response_data["landmarks"] = unity_landmarks
                    response_data["pose_detected"] = True

                    # DEBUG: Save snapshot of key joint positions
                    import json as _json
                    try:
                        _lm = _json.loads(unity_landmarks)
                        _snapshot = {"frame": _vision_count}
                        for j in _lm.get("joints", []):
                            if j["name"] in ("Nose", "RightWrist", "RightKnee"):
                                _snapshot[j["name"]] = {"x": j["x"], "y": j["y"]}
                        _debug_poses.append(_snapshot)
                        if len(_debug_poses) > 20:
                            _debug_poses.pop(0)
                        # Print EVERY frame coordinates
                        nose = _snapshot.get("Nose", {})
                        wrist = _snapshot.get("RightWrist", {})
                        knee = _snapshot.get("RightKnee", {})
                        print(f"🔍 F{_vision_count} Nose=({nose.get('x','?')},{nose.get('y','?')}) Wrist=({wrist.get('x','?')},{wrist.get('y','?')}) Knee=({knee.get('x','?')},{knee.get('y','?')})")
                    except Exception:
                        pass

                    # Process vision for exercise tracking (if in workout)
                    if not cs.get("paused_for_voice", False):
                        vision_feedback = process_vision_data(skeleton_data)
                        if vision_feedback:
                            response_data["feedback"] = vision_feedback

                    response_data["form_status"] = cs.get("form_status", "neutral")
                    response_data["angle"] = round(cs.get("current_angle", 180), 1)
                    response_data["rep_count"] = cs["reps_count"]
        except Exception as e:
            print(f"⚠️ Vision error: {e}")
            import traceback
            traceback.print_exc()

    # Debug: log response occasionally
    if response_data["pose_detected"]:
        angle = response_data.get("angle", 180)
        form = response_data.get("form_status", "?")
        reps = response_data.get("rep_count", 0)
        fb = response_data.get("feedback", "")
        if _vision_count % 5 == 0 or fb:  # ALWAYS log when feedback is given
            print(f"📤 angle={angle}° form={form} reps={reps} fb='{fb}'")

    return JSONResponse(content=response_data)


@app.post("/speak")
async def speak(request: TextRequest):
    """Initial greeting trigger."""
    import json
    import os
    
    # Try to load user name
    user_name = "there"
    try:
        user_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "user_data.json")
        if os.path.exists(user_data_path):
            with open(user_data_path, 'r') as f:
                user_data = json.load(f)
                user_name = user_data.get("user_name", "there")
    except Exception as e:
        print(f"⚠️ Could not load user name for greeting: {e}")

    output = "greeting.mp3"
    session.current_session["state"] = "check_in"
    text = f"Hello {user_name}! Dr. Kineti here. How are you feeling today?"
    session.current_session["last_response"] = text
    await synthesize_speech(text, output)
    return FileResponse(output, media_type="audio/mpeg", filename="greeting.mp3")


@app.post("/stop_workout")
async def stop_workout():
    """Manual trigger to stop the current workout and get a summary."""
    cs = session.current_session
    output = "summary.mp3"
    
    reps = cs.get("reps_count", 0)
    exercise = cs.get("active_exercise", "Squats")
    
    # Update state
    cs["state"] = "finished_workout"
    cs["user_confirmed"] = False
    
    # Save progress
    from app.conversation import get_user_context
    import json, os
    user_data = get_user_context()
    user_data["reps_today"] = user_data.get("reps_today", 0) + reps
    user_data["last_exercise"] = exercise
    try:
        user_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "user_data.json")
        with open(user_data_path, 'w') as f:
            json.dump(user_data, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save user data on stop_workout: {e}")

    # Generate summary speech
    if reps > 0:
        text = f"Workout complete! Excellent job, Swapnil. You finished {reps} {exercise}. Your progress has been saved. Take a rest, and let me know if you want to go again."
    else:
        text = "Workout stopped. Take a breather, Swapnil. Let me know when you're ready to try again."
        
    cs["last_response"] = text
    await synthesize_speech(text, output)
    
    import base64
    with open(output, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    return JSONResponse(content={
        "status": "workout_stopped", 
        "feedback": text,
        "reps": reps,
        "audio_b64": audio_b64
    })


@app.post("/talk")
async def talk(
    file: UploadFile = File(...),
    skeleton_data: str = Form("{}"),
    camera_frame_b64: str = Form("")
):
    """
    Main Loop: Receives Audio + Camera Frame -> Returns JSON with Audio + Landmarks.
    """
    cs = session.current_session
    temp_in = f"temp_{file.filename}"
    output = "response.mp3"

    try:
        response_data = {
            "status": "ok",
            "feedback": "",
            "rep_count": cs["reps_count"],
            "state": cs["state"],
            "landmarks": "{}",
            "audio_b64": "",
            "form_status": cs.get("form_status", "neutral"),
            "angle": round(cs.get("current_angle", 180), 1)
        }

        # ============================================
        # STEP 0: Process Camera Frame with MediaPipe
        # ============================================
        unity_landmarks = "{}"
        if camera_frame_b64 and len(camera_frame_b64) > 100:
            try:
                frame_bytes = base64.b64decode(camera_frame_b64)
                print(f"📷 Frame received: {len(frame_bytes)} bytes")
                result = detect_pose_from_frame(frame_bytes)
                if isinstance(result, tuple):
                    skeleton_data, unity_landmarks = result
                    if unity_landmarks != "{}":
                        print(f"🦴 Detected pose! Landmarks: {len(unity_landmarks)} chars")
                else:
                    skeleton_data = result
            except Exception as e:
                print(f"⚠️ Frame decode error: {e}")
        else:
            if camera_frame_b64:
                print(f"📷 Frame too small: {len(camera_frame_b64)} chars")

        response_data["landmarks"] = unity_landmarks

        # ============================================
        # STEP 1: Check Audio (Skip dummy heartbeat files)
        # ============================================
        with open(temp_in, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(temp_in)
        user_text = ""

        if file_size > MIN_AUDIO_SIZE_BYTES:
            try:
                with open(temp_in, "rb") as audio_file:
                    transcription = config.groq_client.audio.transcriptions.create(
                        file=(temp_in, audio_file.read()),
                        model="whisper-large-v3-turbo",
                        response_format="json",
                        language="en"
                    )
                user_text = transcription.text.strip()
            except Exception as e:
                print(f"⚠️ Transcription error: {e}")
                user_text = ""

        # ============================================
        # STEP 2: If user IS speaking, VOICE takes priority
        # ============================================
        if len(user_text) > 2 and not is_echo(user_text, cs["last_response"]):
            print(f"🗣️ User: {user_text}")

            if cs["state"] == "active_workout":
                cs["paused_for_voice"] = True

            ai_response = manage_conversation(user_text)

            if ai_response:
                cs["last_response"] = ai_response
                print(f"🤖 Coach: {ai_response}")

                await synthesize_speech(ai_response, output)

                cs["paused_for_voice"] = False

                with open(output, "rb") as f:
                    response_data["audio_b64"] = base64.b64encode(f.read()).decode("utf-8")

                response_data["feedback"] = ai_response
                response_data["rep_count"] = cs["reps_count"]
                response_data["state"] = cs["state"]
                response_data["landmarks"] = unity_landmarks

                return JSONResponse(content=response_data)
            else:
                cs["paused_for_voice"] = False

        # ============================================
        # STEP 3: If user is NOT speaking, process Vision
        # ============================================
        if not cs.get("paused_for_voice", False):
            vision_feedback = process_vision_data(skeleton_data)

            if vision_feedback:
                response_data["feedback"] = vision_feedback
                await synthesize_speech(vision_feedback, output)
                with open(output, "rb") as f:
                    response_data["audio_b64"] = base64.b64encode(f.read()).decode("utf-8")
                response_data["landmarks"] = unity_landmarks
                return JSONResponse(content=response_data)

        # ============================================
        # STEP 4: Silent response (still send landmarks!)
        # ============================================
        response_data["status"] = "silent"
        response_data["landmarks"] = unity_landmarks
        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"❌ Server Error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e), "landmarks": "{}"}, status_code=500)
    finally:
        if os.path.exists(temp_in):
            os.remove(temp_in)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)