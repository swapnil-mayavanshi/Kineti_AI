import os
import shutil
import json
import math
import random # <--- Added for random feedback
import soundfile as sf
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
from kokoro import KPipeline 

load_dotenv()
app = FastAPI()

# --- CONFIG ---
PDF_FILE_PATH = "data/acl-protocol.pdf"
USER_DATA_PATH = "user_data.json"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- COACHING DATABASE ---
COACH_PHRASES = {
    "start_squat": [
        "Alright, let's aim for 10 reps. Feet shoulder-width apart. Go.",
        "Position looks good. Give me 10 clean squats. Start when ready.",
        "Okay, Protocol Week {week}. Target depth is {depth} degrees. Let's go."
    ],
    "good_rep": [
        "Perfect.", "Nice control.", "That's it.", "Good form.", "Solid.", "Excellent."
    ],
    "correction_lower": [
        "Go a bit deeper...", "Not quite there, lower...", "Push for that depth...", "Lower..."
    ],
    "correction_posture": [
        "Keep your chest up.", "Control the descent.", "Don't rush it."
    ],
    "complete": [
        "And rest. Great set.", "Target hit. Good work today.", "Relax. You crushed it."
    ]
}

# --- PROTOCOL RULES ---
PROTOCOL_RULES = {
    1: {"max_flexion": 90, "squat_depth": 110, "target_reps": 8, "exercises": ["heel_slide", "leg_raise"]},
    2: {"max_flexion": 100, "squat_depth": 100, "target_reps": 10, "exercises": ["squat"]},
    3: {"max_flexion": 115, "squat_depth": 90, "target_reps": 12, "exercises": ["squat"]},
}

# --- GLOBALS ---
vector_store = None
groq_client = None
llm = None
tts_pipeline = None 

# --- SESSION STATE ---
current_session = {
    "active_exercise": "none", 
    "is_active": False,        
    "reps_session": 0,
    "target_reps": 10,
    "feedback_buffer": "" # Stores "Go Lower" so we don't say it 10 times a second
}

@app.on_event("startup")
def startup_event():
    global vector_store, groq_client, llm, tts_pipeline
    print("⚡ Initializing Dr. Kineti (Coach Persona)...")
    tts_pipeline = KPipeline(lang_code='a') 
    
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0.2)
    
    if os.path.exists(PDF_FILE_PATH):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        loader = PyPDFLoader(PDF_FILE_PATH)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        vector_store = FAISS.from_documents(splits, embeddings)
        print("✅ Brain Ready.")

# --- MEMORY FUNCTIONS ---
def load_history():
    if not os.path.exists(USER_DATA_PATH): return {"current_week": 1, "total_reps": 0}
    with open(USER_DATA_PATH, "r") as f: return json.load(f)

def save_history(data):
    with open(USER_DATA_PATH, "w") as f: json.dump(data, f, indent=4)

# --- MATH TOOLKIT ---
def calculate_angle(p1, p2, p3):
    ax, ay, az = p1['x']-p2['x'], p1['y']-p2['y'], p1['z']-p2['z']
    bx, by, bz = p3['x']-p2['x'], p3['y']-p2['y'], p3['z']-p2['z']
    dot_product = ax*bx + ay*by + az*bz
    mag_a = math.sqrt(ax**2 + ay**2 + az**2)
    mag_b = math.sqrt(bx**2 + by**2 + bz**2)
    if mag_a * mag_b == 0: return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot_product / (mag_a * mag_b)))))

# --- EXERCISE ANALYZERS ---
def analyze_squat(joints, week_data):
    target_depth = week_data["squat_depth"]
    target_reps = week_data.get("target_reps", 10)
    angle = calculate_angle(joints['RightHip'], joints['RightKnee'], joints['RightAnkle'])
    
    print(f"📐 Angle: {int(angle)}° | Target: <{target_depth}")

    # 1. COACHING: Going Down
    if angle < target_depth: 
        if not current_session["is_active"]:
            current_session["is_active"] = True
            return "Good depth. Up." 
    elif angle < (target_depth + 15) and not current_session["is_active"]:
        # They are CLOSE but not deep enough yet
        return random.choice(COACH_PHRASES["correction_lower"])

    # 2. COACHING: Standing Up (Rep Complete)
    elif angle > 160: 
        if current_session["is_active"]:
            current_session["is_active"] = False
            current_session["reps_session"] += 1
            
            reps = current_session["reps_session"]
            
            # Check for Set Completion
            if reps >= target_reps:
                return random.choice(COACH_PHRASES["complete"])
            
            # Random Encouragement every few reps
            phrase = random.choice(COACH_PHRASES["good_rep"])
            return f"{phrase} That is {reps}."
            
    return None

def analyze_movement(skeleton_json):
    global current_session
    try:
        data = json.loads(skeleton_json)
        if "joints" not in data or len(data["joints"]) == 0: return None
        joints = {j['name']: j for j in data['joints']}
        if 'RightKnee' not in joints: return None
        
        hist = load_history()
        week = hist.get("current_week", 1)
        rules = PROTOCOL_RULES.get(week, PROTOCOL_RULES[1])
        
        mode = current_session["active_exercise"]
        
        rep_msg = None
        if mode == "squat": rep_msg = analyze_squat(joints, rules)
        
        if rep_msg:
             # Logic to prevent spamming "Go Lower" every millisecond
             if rep_msg in COACH_PHRASES["correction_lower"]:
                 if current_session["feedback_buffer"] == rep_msg: return None # Don't repeat immediately
                 current_session["feedback_buffer"] = rep_msg
             else:
                 current_session["feedback_buffer"] = "" # Reset on new event
                 
             if any(char.isdigit() for char in rep_msg): # If it counts a rep
                 hist["total_reps"] += 1
                 save_history(hist)
                 
        return rep_msg

    except Exception as e:
        print(f"⚠️ Math Error: {e}")
        return None

# --- VOICE ---
def generate_local_voice(text, output_file):
    try:
        # Sarah is the best "Coach" voice
        generator = tts_pipeline(text, voice='af_sarah', speed=1.2)
        all_audio = []
        for _, _, audio in generator: all_audio.extend(audio)
        sf.write(output_file, all_audio, 24000)
        return True
    except Exception as e: return False

def get_ai_decision(user_text, skeleton_data):
    rep_msg = analyze_movement(skeleton_data)
    if rep_msg: return rep_msg, "none" # Priority: Physical Feedback

    history = load_history()
    week = history.get("current_week", 1)
    rules = PROTOCOL_RULES.get(week, PROTOCOL_RULES[1])

    # TRIGGER EXERCISE MODE
    if "squat" in user_text.lower():
        current_session["active_exercise"] = "squat"
        current_session["reps_session"] = 0
        depth = rules["squat_depth"]
        # Pick a random start phrase
        start_phrase = random.choice(COACH_PHRASES["start_squat"]).format(week=week, depth=depth)
        return start_phrase, "squat"

    # DEFAULT CHAT
    if not vector_store: return "I am ready.", "none"
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    docs = retriever.invoke(user_text)
    context = docs[0].page_content[:600]
    messages = [("system", f"Context: {context}. You are Dr. Kineti. Brief advice."), ("human", user_text)]
    return llm.invoke(messages).content, "none"

# --- ENDPOINTS ---
class TextRequest(BaseModel):
    text: str

@app.post("/speak")
def speak(request: TextRequest):
    output = "greeting.mp3"
    if generate_local_voice("Hello. Dr. Kineti here.", output):
        return FileResponse(output, media_type="audio/mpeg", filename="greeting.mp3")
    raise HTTPException(status_code=500, detail="Audio failed")

@app.post("/talk")
def talk(file: UploadFile = File(...), skeleton_data: str = Form(...)):
    temp_in = f"temp_{file.filename}"
    output = "response.mp3"
    try:
        with open(temp_in, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        
        # 1. Transcribe
        with open(temp_in, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=(temp_in, audio_file.read()), model="whisper-large-v3-turbo", response_format="json", language="en"
            )
        user_text = transcription.text.strip()
        print(f"🗣️ User: {user_text}")

        # 2. Analyze
        rep_msg = analyze_movement(skeleton_data)
        
        headers = {"X-Image-Cmd": "none", "X-Feedback": ""}

        # Priority: Movement Feedback (Even if silent)
        if rep_msg:
            print(f"🏋️ Coach: {rep_msg}")
            headers["X-Feedback"] = rep_msg # Send text to Unity Screen
            if any(char.isdigit() for char in rep_msg): # If it contains a number, it's a rep count
                 # Extract number roughly for the big counter (simple logic)
                 headers["X-Rep-Count"] = ''.join(filter(str.isdigit, rep_msg))

            if generate_local_voice(rep_msg, output):
                return FileResponse(output, media_type="audio/mpeg", filename="response.mp3", headers=headers)

        if len(user_text) < 2: return JSONResponse(content={"status": "silent"}, status_code=200)

        # Normal Chat
        ai_text, image_cmd = get_ai_decision(user_text, skeleton_data)
        headers["X-Image-Cmd"] = image_cmd
        headers["X-Feedback"] = ai_text[:50] + "..." # Show snippet on screen

        if generate_local_voice(ai_text, output):
            return FileResponse(output, media_type="audio/mpeg", filename="response.mp3", headers=headers)
        else: raise HTTPException(status_code=500, detail="Gen Failed")
    finally:
        if os.path.exists(temp_in): os.remove(temp_in)