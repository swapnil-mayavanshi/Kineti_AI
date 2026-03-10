"""
Kineti-AI Conversation Engine
State machine logic for managing the therapy conversation flow.
Includes RAG-powered general fitness Q&A.
"""

import re
import json
import os

from app import config
from app import session
from app.pose_engine import check_dynamic_rule

def get_user_context():
    """Loads user data from JSON to provide personalized context to the LLM."""
    try:
        user_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "user_data.json")
        if os.path.exists(user_data_path):
            with open(user_data_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ Could not load user data: {e}")
    return {"user_name": "Swapnil", "current_day": 1, "reps_today": 0, "last_exercise": "unknown"}

def ask_general_question(user_text):
    """
    Uses RAG + LLM to answer any fitness/PT/health question.
    Falls back to LLM-only if RAG is unavailable.
    """
    if not config.llm:
        return "I'm sorry, I can't answer that right now. Let me know how you're feeling so we can start your exercises!"

    try:
        context = ""
        if config.vector_store:
            retriever = config.vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(user_text)
            context = "\n".join([d.page_content for d in docs]) if docs else ""

        user_data = get_user_context()
        u_name = user_data.get('user_name', 'Swapnil')
        u_day = user_data.get('current_day', 1)
        u_last = user_data.get('last_exercise', 'None recorded')
        u_reps = user_data.get('reps_today', 0)

        prompt = f"""{config.SYSTEM_INSTRUCTION}

Patient Name: {u_name}
Current Therapy Day: {u_day}
Last Exercise Completed: {u_last}
Reps Today: {u_reps}

Context from PT protocol:
{context}

User question/statement: {user_text}

Respond as Dr. Kineti. Be highly personalized, referencing his name and past progress if relevant. Keep it under 3 sentences."""

        response = config.llm.invoke([("system", prompt)]).content
        print(f"🧠 AI answered general question: {response[:80]}...")
        return response.strip()
    except Exception as e:
        print(f"⚠️ General Q&A error: {e}")
        return "That's a great question, Swapnil! I'd recommend discussing it with your physical therapist for personalized advice."


def extract_exercise_rules(exercise_name):
    """Gets the correct angle targets from the RAG/LLM using the Persona."""
    print(f"🔍 AI Reading Protocol for: {exercise_name}...")

    # Default fallback if RAG is off
    default_rules = {"joint": "RightKnee", "target": 90, "mode": "min", "cue": "Control your motion."}
    if not config.vector_store:
        return default_rules

    try:
        retriever = config.vector_store.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke(exercise_name)
        context = docs[0].page_content if docs else ""

        extraction_prompt = f"""
        {config.SYSTEM_INSTRUCTION}
        
        Context: {context}
        Task: Extract biomechanics for '{exercise_name}'.
        Return ONLY valid JSON: {{"joint": "RightKnee", "target": 90, "mode": "min", "cue": "Keep back straight"}}
        """
        response = config.llm.invoke([("system", extraction_prompt)]).content
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception as e:
        print(f"⚠️ Rule Extraction Failed: {e}")

    return default_rules


def _is_question(text):
    """Check if user text looks like a general question."""
    question_starters = ["what", "how", "why", "when", "where", "who", "can",
                         "should", "is", "are", "do", "does", "will", "tell me",
                         "explain", "describe", "help me understand"]
    t = text.lower().strip()
    return any(t.startswith(q) for q in question_starters) or "?" in t


def manage_conversation(user_text):
    """
    Determines AI response based on conversation state.
    Now with general fitness Q&A support via RAG + LLM.
    """
    cs = session.current_session
    user_lower = user_text.lower()
    
    # Load user data for personalization
    user_data = get_user_context()
    u_name = user_data.get("user_name", "Swapnil")
    r_today = user_data.get("reps_today", 0)
    l_ex = user_data.get("last_exercise", "Squat")


    # GLOBAL: Answer rep count questions in ANY state
    rep_question_words = ["how many", "reps", "rep count", "total reps"]
    if any(w in user_lower for w in rep_question_words):
        count = cs['reps_count']
        if count > 0:
            return f"You've done {count} reps! Great job!"
        else:
            return "No reps counted yet. Make sure your full body is visible to the camera when doing squats!"

    # -------------------------------
    # STATE 1: CHECK IN (Assess pain)
    # -------------------------------
    if cs["state"] == "check_in":
        cs["user_confirmed"] = False

        # Fast-track: user directly asks to start exercise → skip propose, go to active_workout
        start_words = ["start", "ready", "begin", "let's go", "exercise", "workout", "squat"]
        if any(w in user_lower for w in start_words):
            cs["pain_level"] = 2
            cs["user_confirmed"] = True
            cs["active_exercise"] = "Squat"
            cs["active_rules"] = extract_exercise_rules("Squat")
            cs["reps_count"] = 0
            cs["is_moving"] = False
            cs["state"] = "active_workout"
            cue = cs["active_rules"].get("cue", "Control your motion.")
            target = cs["active_rules"].get("target", 90)
            return f"Let's do Squats! Aim for {target} degrees. {cue} Begin when you're ready!"

        # Negation words that flip meaning
        negation_words = ["no", "not", "nothing", "dont", "don't", "without", "zero", "none", "never"]
        has_negation = any(neg in user_lower for neg in negation_words)

        pain_words = ["pain", "hurt", "sore", "bad", "terrible", "awful", "worse", "ache"]
        good_words = ["good", "fine", "okay", "great", "better", "amazing", "fantastic", "perfect", "excellent", "well"]
        medium_words = ["medium", "moderate", "so-so", "alright", "okay-ish", "manageable"]

        has_pain_word = any(x in user_lower for x in pain_words)
        has_good_word = any(x in user_lower for x in good_words)
        has_medium_word = any(x in user_lower for x in medium_words)

        # LOGIC: "no pain" / "nothing hurts" = GOOD
        if has_good_word and not (has_pain_word and not has_negation):
            cs["pain_level"] = 2
            cs["state"] = "propose"
            progress_msg = f" You've completed {r_today} reps today! Let's do some more {l_ex}." if r_today > 0 else f" Let's do some {l_ex} today to build strength."
            return f"That's wonderful to hear, {u_name}!{progress_msg} Say 'ready' or 'start' when you want to begin."

        elif has_pain_word and has_negation:
            cs["pain_level"] = 1
            cs["state"] = "propose"
            return f"Great to hear you're pain-free, {u_name}! Let's do some {l_ex}. Say 'ready' when you want to begin."

        elif has_medium_word:
            cs["pain_level"] = 5
            cs["state"] = "propose"
            return f"Thanks for letting me know, {u_name}. Let's take it easy with some Heel Slides. Say 'ready' when you want to start."

        elif has_pain_word and not has_negation:
            cs["pain_level"] = 7
            cs["state"] = "propose"
            return f"I'm sorry to hear that, {u_name}. Since you're experiencing pain, let's try gentle Heel Slides instead of {l_ex}. They're easy on your knee. Just say 'ready' when you want to start."

        # General question during check-in? Answer it!
        elif _is_question(user_text):
            return ask_general_question(user_text)

        else:
            return "Welcome back! How is your pain level today? You can say low, medium, or high."

    # ------------------------------------------------
    # STATE 2: PROPOSE EXERCISE (The Gatekeeper)
    # ------------------------------------------------
    elif cs["state"] == "propose":

        confirmation_phrases = ["yes", "start", "sure", "okay", "go", "ready",
                                "let's do it", "begin", "i'm ready", "let's go",
                                "yep", "yeah", "absolutely", "definitely"]
        rejection_phrases = ["no", "wait", "stop", "not yet", "hold on",
                            "give me a moment", "not ready", "one second"]

        if any(phrase in user_lower for phrase in confirmation_phrases):
            cs["user_confirmed"] = True

            exercise = "Squat"
            if cs["pain_level"] > 4 or "heel" in user_lower:
                exercise = "Heel Slides"

            cs["active_exercise"] = exercise
            cs["active_rules"] = extract_exercise_rules(exercise)
            cs["reps_count"] = 0
            cs["is_moving"] = False
            cs["state"] = "active_workout"

            cue = cs["active_rules"].get("cue", "Control your motion.")
            target = cs["active_rules"].get("target", 90)

            return f"Great! Let's start {exercise}. Aim for {target} degrees. {cue} Begin when you're ready!"

        elif any(phrase in user_lower for phrase in rejection_phrases):
            cs["user_confirmed"] = False
            return "No problem! Take all the time you need. Just say 'ready' whenever you want to start."

        # General question during propose state? Answer via AI!
        elif _is_question(user_text):
            answer = ask_general_question(user_text)
            return answer + " When you're ready to exercise, just say 'start' or 'ready'."

        else:
            return "I need a clear 'yes' or 'ready' before we start. Are you ready to begin the exercise?"

    # -----------------------------------
    # STATE 3: ACTIVE WORKOUT
    # -----------------------------------
    elif cs["state"] == "active_workout":

        if any(x in user_lower for x in ["stop", "done", "finish", "tired", "enough", "break"]):
            reps = cs['reps_count']
            cs["state"] = "finished_workout"  # Go to a resting state, not check_in!
            cs["user_confirmed"] = False
            
            # Save progress to user_data.json
            user_data = get_user_context()
            user_data["reps_today"] = user_data.get("reps_today", 0) + reps
            user_data["last_exercise"] = cs.get("active_exercise", "Squat")
            try:
                user_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "user_data.json")
                with open(user_data_path, 'w') as f:
                    json.dump(user_data, f, indent=2)
            except Exception as e:
                print(f"⚠️ Failed to save user data: {e}")

            if reps > 0:
                return f"Great work, Swapnil! You completed {reps} reps of {cs.get('active_exercise', 'Squats')}. I've saved your progress. Take a breather! Let me know if you have any questions or when you want to continue."
            else:
                return "No problem, we can stop here. Let me know if you need anything else!"

        if any(x in user_lower for x in ["pain", "hurt", "ouch", "sharp"]):
            cs["state"] = "check_in"
            cs["user_confirmed"] = False
            return "Let's stop right there. Your safety comes first. Can you describe where it hurts?"

        # General question during workout? Answer briefly via AI
        if _is_question(user_text):
            # Special handling for rep count questions
            rep_words = ["how many", "reps", "count", "rep"]
            if any(w in user_lower for w in rep_words):
                count = cs['reps_count']
                if count > 0:
                    return f"You've done {count} reps so far. Keep going, you're doing great!"
                else:
                    return "No reps counted yet. Make sure to do full squats — bend your knees deep and stand back up!"
            return ask_general_question(user_text)

        if any(x in user_lower for x in ["am i doing right", "am i doing well", "is this good", "is my form good"]):
            return "Yes! You're doing fantastic. Keep it up!"

        return None  # Let vision handle normal workout flow

    # -----------------------------------
    # STATE 4: FINISHED WORKOUT
    # -----------------------------------
    elif cs["state"] == "finished_workout":
        # If they explicitly ask to start another exercise, loop back to check_in
        start_words = ["start", "another", "more", "continue", "again", "exercise", "workout", "ready"]
        if any(w in user_lower for w in start_words):
            cs["state"] = "check_in"
            return "Awesome! Feeling energized? Let's keep going. How is your pain level right now before we start another set?"
        
        # Otherwise just answer general questions and let them rest
        if _is_question(user_text):
            return ask_general_question(user_text)
            
        return "I'm right here if you need anything. Just say 'start another exercise' when you want to go again!"

    return "I'm here and listening. How can I help you?"

