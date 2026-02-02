import os
import shutil
import json
import math
import random
import asyncio
import edge_tts  # <--- NEW VOICE ENGINE
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- SETUP ---
load_dotenv()
app = FastAPI()

PDF_FILE_PATH = "data/acl-protocol.pdf"
USER_DATA_PATH = "user_data.json"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- 1. VOICE SETTINGS (The "Pro" Vibe) ---
# "en-US-AriaNeural" is the best free 'Doctor' voice available.
VOICE_NAME = "en-US-AriaNeural" 

# --- 2. COACHING PHRASES (Instant Responses) ---
COACH_PHRASES = {
    "start_squat": [
        "Alright, let's aim for 10 reps. Feet shoulder-width apart. Go.",
        "Position looks good. Give me 10 clean squats. Start when ready.",
        "Okay, Protocol Week {week}. Target depth is {depth} degrees. Let's work."
    ],
    "good_rep": [
        "Perfect.", "Nice control.", "That's it.", "Good form.", "Solid.", "Excellent."
    ],
    "correction_lower": [
        "Go a bit deeper.", "Not quite there, lower.", "Push for that depth.", "Lower."
    ],
    "complete": [
        "And rest. Great set.", "Target hit. Good work today.", "Relax. You crushed it."
    ]
}

# --- 3. MEDICAL PROTOCOLS ---
PROTOCOL_RULES = {
    1: {"max_flexion": 90, "squat_depth": 110, "target_reps": 8},
    2: {"max_flexion": 100, "squat_depth": 100, "target_reps": 10},
    3: {"max_flexion": 115, "squat_depth": 90, "target_reps": 12},
}

# --- GLOBALS ---
vector_store = None
groq_client = None
llm = None

# --- SESSION STATE ---
current_session = {
    "active_exercise": "none", 
    "is_active": False,        
    "reps_session": 0,
    "target_reps": 10,
    "feedback_buffer": "",
    "last_response": "" # For Echo Cancellation
}

@app.on_event("startup")
def startup_event():
    global vector_store, groq_client, llm
    print("⚡ Initializing Kineti-AI (Pro Voice + Hybrid Brain)...")
    
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0.2)
    
    # Load Medical Knowledge (RAG)
    if os.path.exists(PDF_FILE_PATH):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        loader = PyPDFLoader(PDF_FILE_PATH)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        vector_store = FAISS.from_documents(splits, embeddings)
        print("✅ Brain Ready.")
    else:
        print("⚠️ Warning: PDF not found. RAG disabled.")

# --- HELPER FUNCTIONS ---
def load_history():
    if not os.path.exists(USER_DATA_PATH): return {"current_week": 1, "total_reps": 0}
    with open(USER_DATA_PATH, "r") as f: return json.load(f)

def save_history(data):
    with open(USER_DATA_PATH, "w") as f: json.dump(data, f, indent=4)

def calculate_angle(p1, p2, p3):
    ax, ay, az = p1['x']-p2['x'], p1['y']-p2['y'], p1['z']-p2['z']
    bx, by, bz = p3['x']-p2['x'], p3['y']-p2['y'], p3['z']-p2['z']
    dot_product = ax*bx + ay*by + az*bz
    mag_a = math.sqrt(ax**2 + ay**2 + az**2)
    mag_b = math.sqrt(bx**2 + by**2 + bz**2)
    if mag_a * mag_b == 0: return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot_product / (mag_a * mag_b)))))

# --- CORE LOGIC: THE MOVEMENT ANALYZER ---
def analyze_movement(skeleton_json):
    global current_session
    try:
        data = json.loads(skeleton_json)
        if "joints" not in data or len(data["joints"]) == 0: return None
        
        # Convert list to dict for easier access
        joints = {j['name']: j for j in data['joints']}
        
        # Start Exercise Check (If user says "Squats", we look for squat motion)
        if current_session["active_exercise"] == "squat" and 'RightKnee' in joints:
            
            # Load Rules
            hist = load_history()
            week = hist.get("current_week", 1)
            rules = PROTOCOL_RULES.get(week, PROTOCOL_RULES[1])
            target_depth = rules["squat_depth"]
            
            # Calculate Angle
            angle = calculate_angle(joints['RightHip'], joints['RightKnee'], joints['RightAnkle'])
            
            # Logic: Down Phase
            if angle < target_depth: 
                if not current_session["is_active"]:
                    current_session["is_active"] = True
                    return "Good depth. Up." 
            elif angle < (target_depth + 15) and not current_session["is_active"]:
                 # Don't spam "Lower" every frame
                 if current_session["feedback_buffer"] != "lower":
                     current_session["feedback_buffer"] = "lower"
                     return random.choice(COACH_PHRASES["correction_lower"])

            # Logic: Up Phase (Rep Complete)
            elif angle > 160: 
                if current_session["is_active"]:
                    current_session["is_active"] = False
                    current_session["reps_session"] += 1
                    reps = current_session["reps_session"]
                    
                    # Update History File
                    hist["total_reps"] += 1
                    save_history(hist)
                    
                    if reps >= rules["target_reps"]: 
                        return random.choice(COACH_PHRASES["complete"])
                    
                    # Hybrid Response: "Good form. 5."
                    phrase = random.choice(COACH_PHRASES["good_rep"])
                    return f"{phrase} {reps}."
            
        return None
    except Exception as e:
        print(f"⚠️ Math Error: {e}")
        return None

# --- NEW VOICE FUNCTION ---
async def generate_pro_voice(text, output_file):
    try:
        communicate = edge_tts.Communicate(text, VOICE_NAME)
        await communicate.save(output_file)
        return True
    except Exception as e:
        print(f"❌ Voice Generation Error: {e}")
        return False

# --- DECISION ENGINE ---
def get_ai_decision(user_text, skeleton_data):
    # 1. Check Movement FIRST (Fastest)
    rep_msg = analyze_movement(skeleton_data)
    if rep_msg: return rep_msg, "none"

    # 2. Check Key Commands
    history = load_history()
    week = history.get("current_week", 1)
    
    if "squat" in user_text.lower():
        current_session["active_exercise"] = "squat"
        current_session["reps_session"] = 0
        depth = PROTOCOL_RULES[week]["squat_depth"]
        start_phrase = random.choice(COACH_PHRASES["start_squat"]).format(week=week, depth=depth)
        return start_phrase, "squat" # Sends "squat" image command

    # 3. Ask the Doctor (LLM)
    if not vector_store: return "I am ready.", "none"
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    docs = retriever.invoke(user_text)
    context = docs[0].page_content[:600]
    
    system_prompt = f"""
    You are Dr. Kineti, a strict but encouraging AI Physical Therapist.
    Context from medical protocol: {context}
    User History: Week {week}.
    Answer briefly (under 2 sentences).
    """
    
    messages = [("system", system_prompt), ("human", user_text)]
    response = llm.invoke(messages).content
    return response, "none"

# --- ENDPOINTS ---
class TextRequest(BaseModel):
    text: str

@app.post("/speak")
async def speak(request: TextRequest):
    # Used for the greeting at app startup
    output = "greeting.mp3"
    text = "Welcome back to Kineti AI. I am ready to start."
    if await generate_pro_voice(text, output):
        return FileResponse(output, media_type="audio/mpeg", filename="greeting.mp3")
    raise HTTPException(status_code=500, detail="Voice failed")

@app.post("/talk")
async def talk(file: UploadFile = File(...), skeleton_data: str = Form(...)):
    temp_in = f"temp_{file.filename}"
    output = "response.mp3"
    
    try:
        # Save audio
        with open(temp_in, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        
        # 1. Transcribe
        with open(temp_in, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=(temp_in, audio_file.read()), model="whisper-large-v3-turbo", response_format="json", language="en"
            )
        user_text = transcription.text.strip()
        print(f"🗣️ User: {user_text}")

        # --- ECHO SAFETY CHECK ---
        # If user text matches the last thing AI said, ignore it.
        last_resp = current_session.get("last_response", "").lower()
        if len(user_text) > 5 and (user_text.lower() in last_resp or last_resp in user_text.lower()):
             print("⚠️ Echo detected. Ignoring.")
             return JSONResponse(content={"status": "echo_ignored"}, status_code=200)

        # 2. Movement Logic (Zero Lag)
        rep_msg = analyze_movement(skeleton_data)
        
        headers = {
            "X-Image-Cmd": "none", 
            "X-Feedback": "", 
            "X-Rep-Count": str(current_session["reps_session"])
        }

        # Priority A: Movement Feedback (Fastest)
        if rep_msg:
            print(f"🏋️ Coach: {rep_msg}")
            headers["X-Feedback"] = rep_msg
            if await generate_pro_voice(rep_msg, output):
                return FileResponse(output, media_type="audio/mpeg", headers=headers)

        # Silent if no input and no movement
        if len(user_text) < 2: 
            return JSONResponse(content={"status": "silent"}, status_code=200)

        # Priority B: AI Conversation
        ai_text, image_cmd = get_ai_decision(user_text, skeleton_data)
        
        # Store response for next echo check
        current_session["last_response"] = ai_text 
        
        headers["X-Image-Cmd"] = image_cmd
        headers["X-Feedback"] = "Listening..." # Clear feedback when chatting

        print(f"🤖 Dr. Kineti: {ai_text}")
        
        if await generate_pro_voice(ai_text, output):
            return FileResponse(output, media_type="audio/mpeg", headers=headers)
        
        raise HTTPException(status_code=500, detail="Gen Failed")

    finally:
        if os.path.exists(temp_in): os.remove(temp_in)

# Run with: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000